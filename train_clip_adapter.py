from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clipadapter.adapter import FeatureAdapter
from clipadapter.clip_model import build_text_features, default_templates, load_openclip
from clipadapter.datasets import get_domain_dataset
from clipadapter.reporting import ensure_report_dirs, append_markdown, format_experiment_markdown


@dataclass
class TrainConfig:
    dataset_root: str
    dataset: str
    source: str
    target: str
    clip_model: str
    clip_pretrained: str
    alpha: float
    bottleneck: int
    lr: float
    batch_size: int
    epochs: int
    num_workers: int
    seed: int
    max_steps: int
    max_eval_batches: int
    precision: str
    device: str
    hf_cache_dir: str


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def evaluate(
    model,
    adapter: FeatureAdapter | None,
    text_features: torch.Tensor,
    loader: DataLoader,
    alpha: float,
    device: torch.device,
    precision: str,
    max_batches: int,
) -> float:
    model.eval()
    if adapter is not None:
        adapter.eval()

    n = 0
    correct = 0

    use_amp = precision == "fp16" and device.type == "cuda"
    for b_idx, (x, y) in enumerate(loader):
        if max_batches > 0 and b_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            f = model.encode_image(x)
        f = f.float()
        f = f / f.norm(dim=-1, keepdim=True)
        if adapter is not None:
            fa = adapter(f)
            f = alpha * fa + (1.0 - alpha) * f
            f = f / f.norm(dim=-1, keepdim=True)
        logit_scale = getattr(model, "logit_scale", None)
        scale = logit_scale.exp().float().clamp(max=100.0) if logit_scale is not None else 100.0
        logits = scale * (f @ text_features.T)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        n += y.numel()
    return correct / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default="/opt/tiger/sahara/ClipAdapter/dataset")
    p.add_argument("--dataset", required=True, choices=["mnist2usps", "svhn2mnist", "office31", "officehome", "pacs"])
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--clip_model", default="ViT-B-32")
    p.add_argument("--clip_pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0, help="0 means full epoch")
    p.add_argument("--max_eval_batches", type=int, default=0, help="0 means full eval")
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--hf_cache_dir", default="/opt/tiger/sahara/ClipAdapter/.cache/hf_datasets")
    p.add_argument(
        "--log_file",
        default="exp.md",
        help="Relative filename under report/ (default: exp.md). Use report.md for final report only.",
    )
    p.add_argument("--exp_name", default="")
    p.add_argument("--save_ckpt", action="store_true")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_paths = ensure_report_dirs(base_dir)
    os.makedirs(args.hf_cache_dir, exist_ok=True)

    log_path = os.path.join(report_paths["report_dir"], args.log_file)

    def _safe_token(s: str) -> str:
        return (
            s.strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

    if args.exp_name:
        exp_name = args.exp_name
    else:
        base = f"{_safe_token(args.source)}2{_safe_token(args.target)}"
        setting = f"lr{args.lr:g}_alpha{args.alpha:g}_b{args.bottleneck}"
        # Avoid duplicated name like mnist2usps_mnist2usps
        if args.dataset.lower() == base.lower():
            exp_name = f"clipadapter_{args.dataset}_{setting}"
        else:
            exp_name = f"clipadapter_{args.dataset}_{base}_{setting}"

    cfg = TrainConfig(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        source=args.source,
        target=args.target,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        alpha=float(args.alpha),
        bottleneck=int(args.bottleneck),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        max_steps=int(args.max_steps),
        max_eval_batches=int(args.max_eval_batches),
        precision=str(args.precision),
        device=str(args.device),
        hf_cache_dir=str(args.hf_cache_dir),
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    bundle = load_openclip(cfg.clip_model, cfg.clip_pretrained, device=device, precision=cfg.precision)
    model = bundle.model

    # dataset routing
    if cfg.dataset == "mnist2usps":
        src_name, tgt_name = "mnist", "usps"
    elif cfg.dataset == "svhn2mnist":
        src_name, tgt_name = "svhn", "mnist"
    elif cfg.dataset == "office31":
        src_name = tgt_name = "office31"
    elif cfg.dataset == "officehome":
        src_name = tgt_name = "officehome"
    elif cfg.dataset == "pacs":
        src_name = tgt_name = "pacs"
    else:
        raise ValueError(cfg.dataset)

    src_split = "train" if src_name in {"mnist", "usps", "svhn"} else "train"
    tgt_split = "test" if tgt_name in {"mnist", "usps", "svhn"} else "all"

    src_ds, classnames = get_domain_dataset(
        dataset_root=cfg.dataset_root,
        dataset_name=src_name,
        domain=cfg.source,
        split=src_split,
        preprocess=bundle.preprocess_train,
        hf_cache_dir=cfg.hf_cache_dir,
        seed=cfg.seed,
    )
    tgt_ds, _ = get_domain_dataset(
        dataset_root=cfg.dataset_root,
        dataset_name=tgt_name,
        domain=cfg.target,
        split=tgt_split,
        preprocess=bundle.preprocess_eval,
        hf_cache_dir=cfg.hf_cache_dir,
        seed=cfg.seed,
    )

    src_loader = DataLoader(
        src_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    tgt_loader = DataLoader(
        tgt_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # text features
    templates = default_templates(src_name)
    text_features = build_text_features(model, bundle.tokenizer, classnames, templates, device=device)
    # Keep classifier (text) features in fp32 for stable matmul / loss.
    text_features = text_features.float()

    # adapter
    with torch.no_grad():
        # infer feature dim from a dummy forward
        x0, _ = next(iter(src_loader))
        x0 = x0.to(device)
        use_amp = cfg.precision == "fp16" and device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            f0 = model.encode_image(x0)
        feat_dim = f0.shape[-1]

    # Keep adapter params in fp32; AMP + FP16 params can break GradScaler.
    adapter = FeatureAdapter(in_dim=feat_dim, bottleneck_dim=cfg.bottleneck).to(device)

    # freeze CLIP
    for p_ in model.parameters():
        p_.requires_grad_(False)
    adapter.train()

    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg.lr)

    use_amp = cfg.precision == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        for x, y in src_loader:
            step += 1
            if cfg.max_steps > 0 and step > cfg.max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                f = model.encode_image(x)
            f = f.float()
            f = f / f.norm(dim=-1, keepdim=True)
            fa = adapter(f)
            f_out = cfg.alpha * fa + (1.0 - cfg.alpha) * f
            f_out = f_out / f_out.norm(dim=-1, keepdim=True)
            logit_scale = getattr(model, "logit_scale", None)
            scale = logit_scale.exp().float().clamp(max=100.0) if logit_scale is not None else 100.0
            logits = scale * (f_out @ text_features.T)
            loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if cfg.max_steps > 0 and step > cfg.max_steps:
            break

    train_time = time.time() - t0

    # eval
    acc_clip = evaluate(model, None, text_features, tgt_loader, cfg.alpha, device, cfg.precision, cfg.max_eval_batches)
    acc_adapter = evaluate(model, adapter, text_features, tgt_loader, cfg.alpha, device, cfg.precision, cfg.max_eval_batches)

    results = {
        "acc_clip_zeroshot": round(acc_clip, 6),
        "acc_clip_adapter": round(acc_adapter, 6),
        "train_time_sec": round(train_time, 2),
        "train_steps": step,
    }

    ckpt_path = ""
    if args.save_ckpt:
        ckpt_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{exp_name}.pt")
        torch.save(
            {
                "exp_name": exp_name,
                "config": asdict(cfg),
                "adapter_state_dict": adapter.state_dict(),
            },
            ckpt_path,
        )
        results["adapter_ckpt"] = ckpt_path

    delta = acc_adapter - acc_clip
    notes = f"Adapter vs. zero-shot: {delta:+.4f} (higher is better)."
    md = format_experiment_markdown(
        exp_name=exp_name,
        dataset=cfg.dataset,
        source=cfg.source,
        target=cfg.target,
        config={
            "clip_model": cfg.clip_model,
            "clip_pretrained": cfg.clip_pretrained,
            "lr": cfg.lr,
            "alpha": cfg.alpha,
            "bottleneck": cfg.bottleneck,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "precision": cfg.precision,
            "seed": cfg.seed,
            "max_steps": cfg.max_steps,
        },
        results=results,
        notes=notes,
    )
    append_markdown(log_path, md)

    print(md)
    if ckpt_path:
        print("saved:", ckpt_path)


if __name__ == "__main__":
    main()

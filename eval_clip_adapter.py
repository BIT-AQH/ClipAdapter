from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from clipadapter.adapter import FeatureAdapter
from clipadapter.clip_model import build_text_features, default_templates, load_openclip
from clipadapter.datasets import get_domain_dataset


@torch.no_grad()
def eval_once(model, adapter, text_features, loader, alpha: float, device: torch.device, precision: str) -> float:
    n = 0
    correct = 0
    use_amp = precision == "fp16" and device.type == "cuda"
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            f = model.encode_image(x)
        f = f.float()
        f = f / f.norm(dim=-1, keepdim=True)
        if adapter is not None:
            fa = adapter(f)
            f = alpha * fa + (1.0 - alpha) * f
            f = f / f.norm(dim=-1, keepdim=True)
        scale = model.logit_scale.exp().float().clamp(max=100.0)
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
    p.add_argument("--ckpt", default="")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--hf_cache_dir", default="/opt/tiger/sahara/ClipAdapter/.cache/hf_datasets")
    args = p.parse_args()

    device = torch.device(args.device)
    bundle = load_openclip(args.clip_model, args.clip_pretrained, device=device, precision=args.precision)
    model = bundle.model

    if args.dataset == "mnist2usps":
        src_name, tgt_name = "mnist", "usps"
    elif args.dataset == "svhn2mnist":
        src_name, tgt_name = "svhn", "mnist"
    elif args.dataset == "office31":
        src_name = tgt_name = "office31"
    elif args.dataset == "officehome":
        src_name = tgt_name = "officehome"
    elif args.dataset == "pacs":
        src_name = tgt_name = "pacs"
    else:
        raise ValueError(args.dataset)

    src_ds, classnames = get_domain_dataset(
        dataset_root=args.dataset_root,
        dataset_name=src_name,
        domain=args.source,
        split="train" if src_name in {"mnist", "usps", "svhn"} else "train",
        preprocess=bundle.preprocess_eval,
        hf_cache_dir=args.hf_cache_dir,
        seed=0,
    )
    tgt_ds, _ = get_domain_dataset(
        dataset_root=args.dataset_root,
        dataset_name=tgt_name,
        domain=args.target,
        split="test" if tgt_name in {"mnist", "usps", "svhn"} else "all",
        preprocess=bundle.preprocess_eval,
        hf_cache_dir=args.hf_cache_dir,
        seed=0,
    )
    loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    templates = default_templates(src_name)
    text_features = build_text_features(model, bundle.tokenizer, classnames, templates, device=device).float()

    # adapter (optional)
    adapter = None
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        # infer feat dim from model output
        with torch.no_grad():
            x0, _ = next(iter(DataLoader(src_ds, batch_size=1)))
            x0 = x0.to(device)
            f0 = model.encode_image(x0)
            feat_dim = int(f0.shape[-1])
        adapter = FeatureAdapter(feat_dim, args.bottleneck).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"], strict=True)
        adapter.eval()

    acc_clip = eval_once(model, None, text_features, loader, args.alpha, device, args.precision)
    acc_adapter = eval_once(model, adapter, text_features, loader, args.alpha, device, args.precision)
    print({"acc_clip_zeroshot": acc_clip, "acc_clip_adapter": acc_adapter})


if __name__ == "__main__":
    main()

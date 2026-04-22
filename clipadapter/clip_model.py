from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch


@dataclass(frozen=True)
class OpenCLIPBundle:
    model: torch.nn.Module
    preprocess_train: Callable
    preprocess_eval: Callable
    tokenizer: Callable


def load_openclip(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str | torch.device = "cuda",
    precision: str = "fp16",
) -> OpenCLIPBundle:
    """Load OpenCLIP model + preprocess + tokenizer.

    precision:
      - "fp16": autocast fp16 on CUDA
      - "fp32": disable autocast
    """
    import open_clip

    model, preprocess_train, preprocess_eval = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    if precision == "fp16" and str(device).startswith("cuda"):
        model.half()
    model.eval()
    return OpenCLIPBundle(
        model=model,
        preprocess_train=preprocess_train,
        preprocess_eval=preprocess_eval,
        tokenizer=tokenizer,
    )


@torch.no_grad()
def build_text_features(
    model: torch.nn.Module,
    tokenizer,
    classnames: List[str],
    templates: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Compute normalized text features (C, D) using prompt ensembling."""
    # Prompt ensemble: average across templates per class
    feats = []
    for name in classnames:
        prompts = [t.format(name) for t in templates]
        tokens = tokenizer(prompts).to(device)
        text_f = model.encode_text(tokens)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        text_f = text_f.mean(dim=0)
        text_f = text_f / text_f.norm()
        feats.append(text_f)
    return torch.stack(feats, dim=0)


def default_templates(dataset_name: str) -> List[str]:
    if dataset_name in {"mnist", "usps", "svhn"}:
        return [
            "a photo of the digit {}.",
            "a centered photo of the digit {}.",
            "a handwritten digit {}.",
        ]
    return [
        "a photo of a {}.",
        "a photo of the {}.",
        "a blurry photo of a {}.",
        "a photo of a small {}.",
        "a photo of a large {}.",
    ]


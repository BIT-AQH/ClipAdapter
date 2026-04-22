from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageListDataset(Dataset):
    """Dataset from an image list file.

    Each line: "relative/path.jpg label".
    """

    def __init__(self, root: str, list_file: str, transform: Optional[Callable] = None):
        self.root = root
        self.list_file = list_file
        self.transform = transform

        raw_samples: List[Tuple[str, int]] = []
        label_to_class_raw: Dict[int, str] = {}

        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, lab = line.split()
                lab_i = int(lab)
                path = os.path.join(root, rel)
                raw_samples.append((path, lab_i))
                # infer class name from path if possible: .../<class>/<file>
                parts = rel.split("/")
                if len(parts) >= 2:
                    cls = parts[-2]
                    label_to_class_raw.setdefault(lab_i, cls)

        # Remap labels to contiguous [0..K-1] to avoid out-of-range targets.
        uniq_labels = sorted({lab for _, lab in raw_samples})
        self.label_map: Dict[int, int] = {lab: i for i, lab in enumerate(uniq_labels)}
        self.samples = [(p, self.label_map[lab]) for (p, lab) in raw_samples]

        # stable class name list (by remapped label id)
        self.classnames = []
        if label_to_class_raw:
            for lab in uniq_labels:
                self.classnames.append(label_to_class_raw.get(lab, str(lab)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


class OfficeHomeDataset(Dataset):
    """Office-Home from local parquet shards (HuggingFace datasets format).

    Stored at: dataset/office-home/data/train-*.parquet
    Columns: image (bytes/path), domain (str), label (ClassLabel)
    """

    def __init__(
        self,
        parquet_files: List[str],
        domain: str,
        transform: Optional[Callable] = None,
        hf_cache_dir: Optional[str] = None,
        split: str = "all",
        split_ratio: float = 0.8,
        seed: int = 0,
    ):
        from datasets import load_dataset

        self.transform = transform

        cache_dir = hf_cache_dir
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "..", ".cache", "hf_datasets")
        os.makedirs(cache_dir, exist_ok=True)

        ds = load_dataset(
            "parquet",
            data_files=parquet_files,
            split="train",
            cache_dir=cache_dir,
        )
        ds = ds.filter(lambda x: x["domain"] == domain)
        self.classnames = ds.features["label"].names

        # deterministic split within the selected domain
        if split in {"train", "val"}:
            n = len(ds)
            idx = list(range(n))
            rnd = random.Random(seed)
            rnd.shuffle(idx)
            cut = int(n * split_ratio)
            idx = idx[:cut] if split == "train" else idx[cut:]
            ds = ds.select(idx)
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[int(idx)]
        img = item["image"]  # already decoded to PIL by datasets
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(item["label"])


def _digits_rgb_transform(preprocess):
    # torchvision digit datasets are often grayscale; convert to RGB first.
    def _t(img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return preprocess(img)

    return _t


def get_domain_dataset(
    dataset_root: str,
    dataset_name: str,
    domain: str,
    split: str,
    preprocess,
    hf_cache_dir: Optional[str] = None,
    seed: int = 0,
) -> Tuple[Dataset, List[str]]:
    """Return (dataset, classnames)."""
    dataset_root = os.path.abspath(dataset_root)
    name = dataset_name.lower()

    if name == "mnist":
        from torchvision.datasets import MNIST

        # Prefer standard torchvision layout: <dataset_root>/MNIST/raw
        if os.path.exists(os.path.join(dataset_root, "MNIST", "raw")):
            root = dataset_root
        else:
            root = os.path.join(dataset_root, "MNIST")
        try:
            ds = MNIST(
                root=root,
                train=(split == "train"),
                download=False,
                transform=_digits_rgb_transform(preprocess),
            )
        except RuntimeError:
            # If only raw files exist (or nothing), let torchvision prepare processed files.
            ds = MNIST(
                root=root,
                train=(split == "train"),
                download=True,
                transform=_digits_rgb_transform(preprocess),
            )
        return ds, [str(i) for i in range(10)]

    if name == "usps":
        from torchvision.datasets import USPS

        root = os.path.join(dataset_root, "usps")
        try:
            ds = USPS(
                root=root,
                train=(split == "train"),
                download=False,
                transform=_digits_rgb_transform(preprocess),
            )
        except RuntimeError:
            ds = USPS(
                root=root,
                train=(split == "train"),
                download=True,
                transform=_digits_rgb_transform(preprocess),
            )
        return ds, [str(i) for i in range(10)]

    if name == "svhn":
        from torchvision.datasets import SVHN

        root = os.path.join(dataset_root, "SVHN")
        split_name = "train" if split == "train" else "test"
        try:
            ds = SVHN(
                root=root,
                split=split_name,
                download=False,
                transform=_digits_rgb_transform(preprocess),
            )
        except RuntimeError:
            ds = SVHN(
                root=root,
                split=split_name,
                download=True,
                transform=_digits_rgb_transform(preprocess),
            )
        return ds, [str(i) for i in range(10)]

    if name == "office31":
        # use provided imagelist for consistent labels
        list_file = os.path.join(dataset_root, "office31", "image_list", f"{domain}.txt")
        ds = ImageListDataset(
            root=os.path.join(dataset_root, "office31"),
            list_file=list_file,
            transform=preprocess,
        )
        return ds, ds.classnames

    if name == "pacs":
        # prefer official splits if present
        split_map = {
            "train": f"{domain}_train.txt",
            "val": f"{domain}_val.txt",
            "all": f"{domain}_all.txt",
        }
        lf = split_map.get(split, split_map["all"])
        list_file = os.path.join(dataset_root, "pacs", "image_list", lf)
        ds = ImageListDataset(
            root=os.path.join(dataset_root, "pacs"),
            list_file=list_file,
            transform=preprocess,
        )
        return ds, ds.classnames

    if name == "officehome":
        files = [
            os.path.join(dataset_root, "office-home", "data", f)
            for f in os.listdir(os.path.join(dataset_root, "office-home", "data"))
            if f.endswith(".parquet")
        ]
        files.sort()
        # split: train/val/all; used for source training; target evaluation uses all
        ds = OfficeHomeDataset(
            parquet_files=files,
            domain=domain,
            transform=preprocess,
            hf_cache_dir=hf_cache_dir,
            split=split,
            seed=seed,
        )
        return ds, ds.classnames

    raise ValueError(f"Unknown dataset_name={dataset_name}")

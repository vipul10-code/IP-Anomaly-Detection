"""
dataset.py
----------
Preprocessing + PyTorch Dataset for the MVTec Anomaly Detection (MVTec AD) format.

Expected folder layout (standard MVTec AD download, unchanged):

    data/mvtec/<category>/train/good/*.png
    data/mvtec/<category>/test/good/*.png
    data/mvtec/<category>/test/<defect_type>/*.png
    data/mvtec/<category>/ground_truth/<defect_type>/*_mask.png

Only "good" (defect-free) images are used for TRAINING, which follows the
standard unsupervised anomaly-detection protocol used for MVTec AD: the model
never sees a defect during training, and has to learn what "normal" leather /
transistor texture looks like well enough that anything else stands out as a
reconstruction error at test time.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    path: str
    label: int              # 0 = normal, 1 = anomalous
    defect_type: str        # "good" or e.g. "cut", "fold", "bent_lead" ...
    mask_path: Optional[str]  # ground-truth pixel mask, only for defect_type != "good"


def scan_split(root_dir: str, category: str, split: str) -> List[Sample]:
    """Walk a train/ or test/ split and build the sample list with labels.

    label = 0 for the 'good' subfolder, 1 for every defect subfolder.
    For split == 'test', mask_path is resolved from the parallel
    ground_truth/<defect_type>/<same_stem>_mask.png file when it exists.
    """
    cat_dir = Path(root_dir) / category / split
    if not cat_dir.exists():
        raise FileNotFoundError(
            f"Expected MVTec split folder not found: {cat_dir}\n"
            f"Download MVTec AD, extract it, and point data.root_dir in config.yaml "
            f"at the folder that contains '<category>/train' and '<category>/test'."
        )

    samples: List[Sample] = []
    for defect_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
        label = 0 if defect_dir.name == "good" else 1
        for img_path in sorted(list(defect_dir.glob("*.png")) + list(defect_dir.glob("*.jpg"))):
            mask_path = None
            if label == 1:
                candidate = (
                    Path(root_dir) / category / "ground_truth" / defect_dir.name
                    / f"{img_path.stem}_mask.png"
                )
                mask_path = str(candidate) if candidate.exists() else None
            samples.append(
                Sample(
                    path=str(img_path),
                    label=label,
                    defect_type=defect_dir.name,
                    mask_path=mask_path,
                )
            )
    return samples


def _load_image(path: str, img_size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR, uint8
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img


def _load_mask(path: Optional[str], img_size: int) -> np.ndarray:
    if path is None:
        return np.zeros((img_size, img_size), dtype=np.float32)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


class MVTecDataset(Dataset):
    """
    split="train" -> only normal images, augmentation optionally applied.
    split="test"  -> every image (good + all defect types), no augmentation,
                     each item also carries its ground-truth mask and defect type.
    """

    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str,
        img_size: int = 256,
        augment_cfg: Optional[dict] = None,
    ):
        self.category = category
        self.split = split
        self.img_size = img_size
        self.augment_cfg = augment_cfg or {}
        self.samples = scan_split(root_dir, category, split)
        if split == "train":
            self.samples = [s for s in self.samples if s.label == 0]
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for {category}/{split}")

    def __len__(self):
        return len(self.samples)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        cfg = self.augment_cfg
        if cfg.get("flip") and np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
        if cfg.get("flip") and np.random.rand() < 0.5:
            img = cv2.flip(img, 0)
        if cfg.get("rotate") and np.random.rand() < 0.5:
            k = np.random.choice([1, 2, 3])
            img = np.rot90(img, k).copy()
        if cfg.get("color_jitter") and np.random.rand() < 0.5:
            factor = np.random.uniform(0.85, 1.15)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = _load_image(s.path, self.img_size)

        if self.split == "train":
            img = self._augment(img)

        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)  # CHW, [0,1]

        if self.split == "train":
            return img_t

        mask = _load_mask(s.mask_path, self.img_size)
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # 1HW
        return {
            "image": img_t,
            "mask": mask_t,
            "label": s.label,
            "defect_type": s.defect_type,
            "path": s.path,
        }


def test_collate(batch):
    """Custom collate for the test split since it returns a dict with strings."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
        "defect_type": [b["defect_type"] for b in batch],
        "path": [b["path"] for b in batch],
    }

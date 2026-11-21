"""
run_patchcore.py
----------------
Training, evaluation, and inference runner for PatchCore on MVTec AD.
Uses the shared zero-leakage evaluation layer (src/evaluator.py).

Usage:
    python -m src.run_patchcore --category leather --mode all
    python -m src.run_patchcore --category transistor --mode all
"""

import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset import MVTecDataset
from src.models.patchcore import PatchCore
from src.evaluator import evaluate_model
from src.utils import get_device, set_seed


def run_patchcore(cfg: dict, category: str, mode: str = "all", target_coreset_size: int = 1500):
    set_seed(cfg.get("train", {}).get("seed", 42))
    device = get_device(cfg.get("train", {}).get("device", "cuda"))
    d_cfg = cfg["data"]
    ckpt_dir = Path(cfg.get("train", {}).get("checkpoint_dir", "./outputs/checkpoints"))
    ckpt_path = ckpt_dir / f"patchcore_{category}.pt"

    model = PatchCore(
        backbone_name=cfg.get("patchcore", {}).get("backbone", "wide_resnet50_2"),
        target_coreset_size=target_coreset_size,
        device=device,
    )

    if mode in ("train", "all") or not ckpt_path.exists():
        train_ds = MVTecDataset(
            root_dir=d_cfg["root_dir"],
            category=category,
            split="train",
            img_size=d_cfg.get("img_size", 256),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=d_cfg.get("batch_size", 16),
            shuffle=False,
            num_workers=0,
        )
        model.fit(train_loader)
        model.save_checkpoint(str(ckpt_path), extra_meta={"category": category})
    else:
        model.load_checkpoint(str(ckpt_path))

    if mode in ("evaluate", "all"):
        print(f"\n[PatchCore][{category}] Running shared zero-leakage evaluation...")
        summary = evaluate_model(
            model_name="patchcore",
            category=category,
            cfg=cfg,
            predict_fn=model.predict,
            device=device,
        )
        return summary


def main():
    parser = argparse.ArgumentParser(description="PatchCore MVTec AD Runner")
    parser.add_argument("--category", required=True, choices=["leather", "transistor"])
    parser.add_argument("--mode", default="all", choices=["train", "evaluate", "all"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--coreset-size", type=int, default=1500)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_patchcore(cfg, args.category, mode=args.mode, target_coreset_size=args.coreset_size)


if __name__ == "__main__":
    main()

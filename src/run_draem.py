"""
run_draem.py
------------
Training and evaluation runner for DRAEM on the leather category.
Uses the shared zero-leakage evaluation layer (src/evaluator.py).

Usage:
    python -m src.run_draem --category leather --mode all --epochs 10
"""

import argparse
import time
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset import MVTecDataset
from src.models.draem import DRAEM
from src.evaluator import evaluate_model
from src.utils import get_device, set_seed


def run_draem(cfg: dict, category: str = "leather", mode: str = "all", epochs: int = 10, lr: float = 1e-4):
    set_seed(cfg.get("train", {}).get("seed", 42))
    device = get_device(cfg.get("train", {}).get("device", "cuda"))
    d_cfg = cfg["data"]
    ckpt_dir = Path(cfg.get("train", {}).get("checkpoint_dir", "./outputs/checkpoints"))
    ckpt_path = ckpt_dir / f"draem_{category}.pt"

    model = DRAEM(device=device)

    if mode in ("train", "all") or not ckpt_path.exists():
        train_ds = MVTecDataset(
            root_dir=d_cfg["root_dir"],
            category=category,
            split="train",
            img_size=d_cfg.get("img_size", 256),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=8,
            shuffle=True,
            num_workers=0,
        )

        opt_recon = torch.optim.Adam(model.reconstructor.parameters(), lr=lr)
        opt_seg = torch.optim.Adam(model.segmentor.parameters(), lr=lr)

        print(f"[DRAEM][{category}] Training for {epochs} epochs...")
        for ep in range(1, epochs + 1):
            t0 = time.time()
            running_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                loss_dict = model.train_step(batch, opt_recon, opt_seg)
                running_loss += loss_dict["total_loss"]
                n_batches += 1
            avg_loss = running_loss / max(1, n_batches)
            print(f"[DRAEM][{category}] Epoch {ep:02d}/{epochs:02d} - Loss: {avg_loss:.4f} ({time.time() - t0:.1f}s)")

        model.save_checkpoint(str(ckpt_path), extra_meta={"category": category, "epochs": epochs})
    else:
        model.load_checkpoint(str(ckpt_path))

    if mode in ("evaluate", "all"):
        print(f"\n[DRAEM][{category}] Running shared zero-leakage evaluation...")
        summary = evaluate_model(
            model_name="draem",
            category=category,
            cfg=cfg,
            predict_fn=model.predict,
            device=device,
        )
        return summary


def main():
    parser = argparse.ArgumentParser(description="DRAEM Leather Runner")
    parser.add_argument("--category", default="leather", choices=["leather"])
    parser.add_argument("--mode", default="all", choices=["train", "evaluate", "all"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_draem(cfg, args.category, mode=args.mode, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()

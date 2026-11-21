"""
train.py
--------
Trains one ConvAutoencoder per MVTec category (leather / transistor) on the
"good" (defect-free) training images only.

Usage:
    python main.py train --category leather
    python main.py train --category transistor
"""

import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import MVTecDataset
from src.model import build_model
from src.losses import ReconstructionLoss
from src.utils import set_seed, get_device


def train_one_category(cfg: dict, category: str):
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["train"]["device"])
    print(f"[{category}] using device: {device}")

    d_cfg = cfg["data"]
    augment_cfg = d_cfg.get("augment", {}).get(category, {})

    train_ds = MVTecDataset(
        root_dir=d_cfg["root_dir"],
        category=category,
        split="train",
        img_size=d_cfg["img_size"],
        augment_cfg=augment_cfg,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=d_cfg["batch_size"],
        shuffle=True,
        num_workers=d_cfg["num_workers"],
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[{category}] training samples (good only): {len(train_ds)}")

    model = build_model(cfg).to(device)
    criterion = ReconstructionLoss(
        ssim_weight=cfg["train"]["ssim_weight"], channels=d_cfg["channels"]
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    log_dir = Path(cfg["train"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    patience_left = cfg["train"]["early_stop_patience"]
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        epoch_start = time.time()
        running = {"total": 0.0, "ssim_loss": 0.0, "mse_loss": 0.0}

        for batch in train_loader:
            imgs = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            recon = model(imgs)
            loss, parts = criterion(recon, imgs)
            loss.backward()
            optimizer.step()

            running["total"] += loss.item()
            running["ssim_loss"] += parts["ssim_loss"]
            running["mse_loss"] += parts["mse_loss"]

        n_batches = len(train_loader)
        for k in running:
            running[k] /= max(1, n_batches)

        scheduler.step(running["total"])
        elapsed = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[{category}] epoch {epoch:03d}/{cfg['train']['epochs']} "
            f"loss={running['total']:.4f} (ssim={running['ssim_loss']:.4f}, "
            f"mse={running['mse_loss']:.4f}) lr={lr_now:.2e} time={elapsed:.1f}s"
        )
        history.append({"epoch": epoch, "lr": lr_now, **running})

        if running["total"] < best_loss - 1e-5:
            best_loss = running["total"]
            patience_left = cfg["train"]["early_stop_patience"]
            torch.save(
                {"model_state": model.state_dict(), "cfg": cfg, "epoch": epoch, "loss": best_loss},
                ckpt_dir / f"{category}_best.pt",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{category}] early stopping at epoch {epoch} (best loss {best_loss:.4f})")
                break

    torch.save(
        {"model_state": model.state_dict(), "cfg": cfg, "epoch": epoch, "loss": running["total"]},
        ckpt_dir / f"{category}_last.pt",
    )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(log_dir / f"{category}_train_history.csv", index=False)

    _plot_loss_curve(hist_df, log_dir / f"{category}_loss_curve.png", category)
    print(f"[{category}] training complete. Best loss: {best_loss:.4f}")
    return ckpt_dir / f"{category}_best.pt"


def _plot_loss_curve(hist_df: pd.DataFrame, out_path: Path, category: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(hist_df["epoch"], hist_df["total"], label="total loss")
    plt.plot(hist_df["epoch"], hist_df["ssim_loss"], label="ssim loss", alpha=0.7)
    plt.plot(hist_df["epoch"], hist_df["mse_loss"], label="mse loss", alpha=0.7)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Training loss — {category}")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

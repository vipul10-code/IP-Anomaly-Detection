"""
main.py
-------
CLI entry point.

    python main.py train    --category leather
    python main.py train    --category transistor
    python main.py evaluate --category leather
    python main.py evaluate --category transistor
    python main.py all      --category leather      # train then evaluate

Config is read from config.yaml (edit paths/hyperparameters there).
"""

import argparse
import yaml

from src.train import train_one_category
from src.evaluate import evaluate_category
from src.infer import predict_image
from src.benchmark import run_benchmark

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="MVTec Anomaly Detection Pipeline & Benchmark")
    parser.add_argument("mode", choices=["train", "evaluate", "all", "predict", "benchmark"])
    parser.add_argument("--category", choices=["leather", "transistor"], help="Category (required for train/eval/predict)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--image", help="Path to a single image file (required for 'predict')")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "benchmark":
        run_benchmark(cfg)
        return

    if not args.category:
        parser.error(f"--category is required for mode '{args.mode}'")

    if args.mode in ("train", "all"):
        train_one_category(cfg, args.category)
    if args.mode in ("evaluate", "all"):
        evaluate_category(cfg, args.category)
    if args.mode == "predict":
        if not args.image:
            parser.error("--image is required for 'predict' mode")
        predict_image(cfg, args.category, args.image)


if __name__ == "__main__":
    main()

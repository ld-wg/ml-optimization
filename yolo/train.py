
#!/usr/bin/env python3
"""
WIDER FACE → YOLOv8 trainer (minimal & tidy)
- Keeps your existing dataset prep/validation in utils.dataset
- Removes custom optimizer/hardware plumbing
- Fixes indentation bug and mistaken sys.exit
- Small, readable, and safe on CPU/MPS/CUDA

Usage (examples):
  python train.py --fraction 0.01 --epochs 3 --batch-size 8 --workers 4
  python train.py --fraction 1.0  --epochs 50 --batch-size 16 --workers 8 --imgsz 640
"""

from __future__ import annotations
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

import yaml

# --- Local utils (dataset only; keep minimal dependencies) ---
from utils.dataset import (
    prepare_dataset,
    check_dataset_integrity,
    validate_yolo_labels,
)

# =====================
# Configuration
# =====================
DEFAULT_FRACTION = 0.01
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 4
DEFAULT_IMGSZ = 640

DATA_PATH = Path("../widerface").resolve()
TRAIN_ANNOTATIONS = DATA_PATH / "wider_face_split" / "wider_face_train_bbx_gt.txt"
VAL_ANNOTATIONS   = DATA_PATH / "wider_face_split" / "wider_face_val_bbx_gt.txt"
TRAIN_IMAGES_SRC  = DATA_PATH / "WIDER_train" / "images"
VAL_IMAGES_SRC    = DATA_PATH / "WIDER_val"   / "images"

DATASET_DIR = Path("./datasets/wider_face_yolo").resolve()
MODEL_FILE = "yolov8n.pt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def pick_device() -> str:
    """Prefer CUDA → MPS → CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "0"  # first CUDA device
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def write_dataset_yaml(yaml_path: Path, dataset_dir: Path) -> None:
    data = {
        "train": str(dataset_dir / "images" / "train"),
        "val":   str(dataset_dir / "images" / "val"),
        "nc": 1,
        "names": ["face"],  # list form is accepted by Ultralytics
    }
    yaml_path.write_text(yaml.dump(data, sort_keys=False))


def train_model(*, epochs: int, batch_size: int, workers: int, imgsz: int) -> Dict[str, Any]:
    """Run a compact YOLOv8 training with conservative, stable settings."""
    device = pick_device()
    logger.info(f"Training on device: {device}")

    # Create dataset YAML
    yaml_path = Path("wider_face.yaml")
    write_dataset_yaml(yaml_path, DATASET_DIR)

    # Experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = f"train_{epochs}ep_{timestamp}"

    # Load model
    model = YOLO(MODEL_FILE)
    logger.info(f"Starting training: {experiment}")
    logger.info(f"Dataset YAML: {yaml_path}")

    # Conservative/portable training args for CPU/MPS/CUDA
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        device=device,
        project="runs/train",
        name=experiment,
        exist_ok=True,
        optimizer="auto",
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        rect=False,
        save_period=1,
        patience=10,
        cos_lr=True,
        deterministic=True,
        seed=42,
    )

    # Locate best weights
    model_path = Path("runs/train") / experiment / "weights" / "best.pt"
    if not model_path.exists():
        logger.error("Training failed - best.pt not found")
        return {"success": False}

    logger.info(f"Training complete → {model_path}")

    # Validate
    val_results = YOLO(str(model_path)).val(data=str(yaml_path))
    try:
        map50 = getattr(getattr(val_results, "box", None), "map50", 0.0) or 0.0
    except Exception:
        map50 = 0.0
    logger.info(f"Validation mAP50: {map50:.3f}")

    return {
        "success": True,
        "experiment": experiment,
        "model_path": str(model_path),
        "val_map50": map50,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train YOLOv8 on WIDER FACE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fraction", type=float, default=DEFAULT_FRACTION, help="Dataset fraction to use")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Data loader workers")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Image size")
    args = p.parse_args()

    logger.info("=" * 64)
    logger.info("WIDER FACE YOLOv8 Training (minimal)")
    logger.info("=" * 64)
    logger.info(f"Fraction={args.fraction} Epochs={args.epochs} Batch={args.batch_size} Workers={args.workers} ImgSz={args.imgsz}")

    # --- Required input paths sanity ---
    required = [
        (DATA_PATH, "WIDER FACE data directory"),
        (TRAIN_ANNOTATIONS, "Train annotations"),
        (VAL_ANNOTATIONS, "Val annotations"),
        (TRAIN_IMAGES_SRC, "Train images"),
        (VAL_IMAGES_SRC, "Val images"),
    ]
    missing = [desc for path, desc in required if not path.exists()]
    if missing:
        for path, desc in required:
            if not path.exists():
                logger.error(f"Missing: {desc} at {path}")
        sys.exit(1)

    # --- Prepare dataset (keeps your existing converter) ---
    try:
        stats = prepare_dataset(args.fraction, DATA_PATH, DATASET_DIR)
        if stats.get("train_count", 0) == 0 or stats.get("val_count", 0) == 0:
            logger.error("Dataset preparation returned zero images.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Dataset preparation error: {e}")
        sys.exit(1)

    # --- Validate pairs & labels quickly ---
    logger.info("Validating prepared dataset…")
    train_ok = check_dataset_integrity(DATASET_DIR / "images" / "train", DATASET_DIR / "labels" / "train")
    val_ok   = check_dataset_integrity(DATASET_DIR / "images" / "val",   DATASET_DIR / "labels" / "val")

    critical = (
        len(train_ok.get("image_without_label", [])) > 0
        or len(train_ok.get("label_without_image", [])) > 0
        or len(val_ok.get("image_without_label", [])) > 0
        or len(val_ok.get("label_without_image", [])) > 0
    )
    if critical:
        logger.error("Critical dataset integrity issues found; aborting.")
        logger.error(f"Train issues: {train_ok}")
        logger.error(f"Val issues:   {val_ok}")
        sys.exit(1)

    tval = validate_yolo_labels(DATASET_DIR / "labels" / "train")
    vval = validate_yolo_labels(DATASET_DIR / "labels" / "val")
    if tval.get("valid_files", 0) == 0 or vval.get("valid_files", 0) == 0:
        logger.error("No valid label files after preparation; aborting.")
        logger.error(f"Train: {tval}")
        logger.error(f"Val:   {vval}")
        sys.exit(1)

    logger.info("Dataset validation passed ✓")

    # --- Train ---
    try:
        results = train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            imgsz=args.imgsz,
        )
        if not results.get("success"):
            sys.exit(1)
        logger.info("=" * 64)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"  Experiment: {results['experiment']}")
        logger.info(f"  Model:      {results['model_path']}")
        logger.info(f"  mAP50:      {results['val_map50']:.3f}")
        logger.info("=" * 64)
    except Exception as e:
        logger.error(f"Training error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

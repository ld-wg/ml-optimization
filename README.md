# YOLOv8 Face Detection Optimization

Training YOLOv8 models on the WIDER FACE dataset and optimizing parameters like learning rate, optimizer, etc.

## Research Paper

- [What is YOLOv8: An In-Depth Exploration of the Internal Features of the Next-Generation Object Detector](https://arxiv.org/html/2408.15857)

## Dataset

- **WIDER FACE**: http://shuoyang1213.me/WIDERFACE/
- **Paper**: [WIDER FACE: A Face Detection Benchmark](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_WIDER_FACE_A_CVPR_2016_paper.pdf)

## What We're Doing

- Train YOLOv8 on WIDER FACE dataset
- Optimize hyperparameters (learning rate, optimizer, etc.)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

```bash
# Quick test with 1% of data, 5 epochs
python train.py --fraction 0.01 --epochs 5
```

## Configuration

The framework uses 4 main parameters (all configurable via command line):

- `--fraction`: Dataset fraction to use (0.0-1.0, default: 0.01)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 16)
- `--workers`: Number of data loader workers (default: 4)

Additional options:

- `--optimizer`: Built-in optimizer ('auto', 'Adam', 'AdamW', 'SGD')
- `--custom-optimizer`: Custom optimizer module:class

## File Structure

```
yolo/
├── train.py              # Main training script (simplified)
├── webcam.py             # Webcam inference script
├── custom_optimizers.py  # Custom optimizer implementations
├── utils/
│   ├── dataset.py        # Dataset preparation and validation
│   ├── hardware.py       # Hardware detection and optimization
│   └── optimizer.py      # Custom optimizer support
├── requirements.txt      # Dependencies
├── wider_face.yaml       # Auto-generated dataset config
└── yolov8n.pt           # YOLOv8 nano model weights

datasets/                # Auto-created during training
└── wider_face_yolo/
    ├── images/
    │   ├── train/       # Symlinked training images
    │   └── val/         # Symlinked validation images
    └── labels/
        ├── train/       # YOLO format labels
        └── val/         # YOLO format labels

runs/                    # Auto-created during training
└── train/
    └── train_XXep_TIMESTAMP/
        ├── weights/
        │   └── best.pt  # Best model checkpoint
        └── ...          # Training logs and metrics
```

## What the Framework Does

1. **Dataset Preparation** (automatic):

   - Converts WIDER FACE annotations to YOLO format
   - Creates symlinks to images for efficient data access
   - Ensures perfect pairing between images and labels
   - Samples dataset according to specified fraction

2. **Training**:

   - Trains YOLOv8 nano model on prepared dataset
   - Supports GPU acceleration (CUDA, MPS) and CPU
   - Built-in or custom optimizers
   - Automatic validation after training

3. **Output**:
   - Saves best model to `runs/train/<experiment>/weights/best.pt`
   - Generates training metrics and curves
   - Reports validation mAP50 score

## Dataset Requirements

Expected directory structure:

```
../widerface/
├── wider_face_split/
│   ├── wider_face_train_bbx_gt.txt
│   └── wider_face_val_bbx_gt.txt
├── WIDER_train/
│   └── images/
└── WIDER_val/
    └── images/
```

## Examples

### Full Training (100% data, 50 epochs)

```bash
python train.py --fraction 1.0 --epochs 50 --batch-size 32 --workers 8
```

### With Custom Optimizer

```bash
python train.py --fraction 0.1 --epochs 20 \
    --custom-optimizer custom_optimizers:AdamWLookahead
```

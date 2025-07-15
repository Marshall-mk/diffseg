# Diffusion Segmentation

A PyTorch implementation of a diffusion-based image segmentation model that uses denoising diffusion probabilistic models (DDPM) for generating segmentation masks.

## Features

- **Diffusion-based segmentation**: Uses forward and reverse diffusion processes for mask generation
- **UNet backbone**: ResNet blocks with attention mechanisms for robust feature extraction
- **Flexible architecture**: Supports different image sizes and number of classes
- **Training utilities**: Complete training pipeline with checkpointing and visualization
- **Inference tools**: Batch processing, interactive mode, and single image inference
- **Synthetic data support**: Built-in synthetic data generation for testing

## Repository Structure

```
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── blocks.py          # Neural network building blocks
│   │   ├── unet.py           # UNet architecture
│   │   └── diffusion_model.py # Main diffusion model
│   ├── training/             # Training utilities (future)
│   └── inference/            # Inference utilities (future)
├── utils/
│   ├── __init__.py
│   ├── data_utils.py         # Data loading and preprocessing
│   └── visualization.py     # Visualization utilities
├── train.py                  # Training script
├── inference.py              # Inference script
├── demo.py                   # Demo and testing script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diffusion-segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Demo
Run the demo to see the model in action with synthetic data:
```bash
python demo.py
```

### Training
Train the model with synthetic data:
```bash
python train.py --synthetic --epochs 50 --batch-size 8
```

Train with your own dataset:
```bash
python train.py --image-dir /path/to/images --mask-dir /path/to/masks --epochs 100
```

### Inference
Run inference on a single image:
```bash
python inference.py --checkpoint /path/to/checkpoint.pth --input /path/to/image.jpg
```

Batch inference on a directory:
```bash
python inference.py --checkpoint /path/to/checkpoint.pth --input /path/to/images/ --batch --output /path/to/results/
```

Interactive inference:
```bash
python inference.py --checkpoint /path/to/checkpoint.pth --interactive
```

## Model Architecture

The model consists of:

1. **Forward Diffusion Process**: Gradually adds noise to ground truth masks
2. **UNet Architecture**: 
   - Encoder-decoder structure with skip connections
   - ResNet blocks with time embeddings
   - Self-attention mechanisms in the middle layers
   - Sinusoidal position embeddings for timestep encoding
3. **Reverse Diffusion Process**: Denoises from pure noise to clean segmentation masks

## Training

### Command Line Arguments

- `--data-dir`: Path to dataset directory
- `--image-dir`: Path to images directory  
- `--mask-dir`: Path to masks directory
- `--output-dir`: Output directory (default: ./outputs)
- `--batch-size`: Batch size (default: 8)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--image-size`: Image size (default: 256)
- `--timesteps`: Number of diffusion timesteps (default: 1000)
- `--synthetic`: Use synthetic data for testing
- `--wandb`: Enable Weights & Biases logging
- `--resume`: Path to checkpoint to resume from

### Example Training Commands

```bash
# Quick test with synthetic data
python train.py --synthetic --epochs 10 --batch-size 4

# Full training with real data
python train.py --image-dir ./data/images --mask-dir ./data/masks --epochs 100 --wandb

# Resume training from checkpoint
python train.py --resume ./outputs/checkpoint_epoch_50.pth --epochs 100
```

## Inference

### Command Line Arguments

- `--checkpoint`: Path to model checkpoint (required)
- `--input`: Input image path or directory
- `--output`: Output directory (default: ./inference_results)
- `--batch`: Batch inference mode for directories
- `--interactive`: Interactive inference mode
- `--steps`: Number of inference steps (default: 50)
- `--image-size`: Image size for processing (default: 256)
- `--visualize-process`: Visualize the inference process

### Example Inference Commands

```bash
# Single image
python inference.py --checkpoint model.pth --input image.jpg

# Batch processing
python inference.py --checkpoint model.pth --input ./images/ --batch --output ./results/

# Interactive mode
python inference.py --checkpoint model.pth --interactive

# Visualize inference process
python inference.py --checkpoint model.pth --input image.jpg --visualize-process
```

## Custom Dataset Setup

### Data Format Requirements

The model expects:
- **Images**: RGB images in common formats (JPG, JPEG, PNG, BMP, TIFF)
- **Masks**: Grayscale masks (0-255, where 255 represents foreground/object)
- **Image-Mask Correspondence**: Each image must have a corresponding mask with matching filename

### Directory Structure Options

#### Option 1: Separate Directories (Recommended)
```
your_dataset/
├── images/
│   ├── image001.jpg
│   ├── image002.png
│   └── ...
└── masks/
    ├── image001.png      # Same name as image
    ├── image002.png
    └── ...
```

#### Option 2: Alternative Mask Naming
```
your_dataset/
├── images/
│   ├── photo_001.jpg
│   ├── photo_002.jpg
│   └── ...
└── masks/
    ├── photo_001_mask.png    # With "_mask" suffix
    ├── photo_002_gt.png      # With "_gt" suffix  
    └── ...
```

#### Option 3: Train/Val Pre-split
```
your_dataset/
├── train/
│   ├── images/
│   │   ├── train_001.jpg
│   │   └── ...
│   └── masks/
│       ├── train_001.png
│       └── ...
└── val/
    ├── images/
    │   ├── val_001.jpg
    │   └── ...
    └── masks/
        ├── val_001.png
        └── ...
```

### Data Preparation Tools

#### Analyze Your Dataset
```bash
# Get detailed statistics about your dataset
python prepare_data.py --action analyze --image-dir ./data/images --mask-dir ./data/masks --output-dir ./analysis

# This will show:
# - Number of images and masks
# - Image size statistics
# - Common issues and recommendations
```

#### Validate Dataset
```bash
# Check for common issues
python prepare_data.py --action validate --image-dir ./data/images --mask-dir ./data/masks

# Automatically fix simple issues
python prepare_data.py --action validate --image-dir ./data/images --mask-dir ./data/masks --fix-issues
```

#### Organize Dataset
```bash
# Automatically split into train/val
python prepare_data.py --action organize --source-dir ./raw_data --output-dir ./organized_data --train-split 0.8
```

#### Create Sample Dataset
```bash
# Generate synthetic data for testing
python prepare_data.py --action create-sample --output-dir ./sample_data --num-samples 50
```

### Data Loading & Augmentation

#### Basic Loading
```python
from utils.data_utils import load_dataset

# Simple loading
train_loader = load_dataset(
    image_dir="./data/images",
    mask_dir="./data/masks",
    batch_size=8,
    image_size=(256, 256)
)
```

#### With Validation Split
```python
# Automatic train/val split
train_loader, val_loader = load_dataset(
    image_dir="./data/images",
    mask_dir="./data/masks",
    batch_size=8,
    val_split=0.2,  # 20% for validation
    augmentation_mode="medium"
)
```

#### Advanced Configuration
```python
# Full control over data loading
train_loader = load_dataset(
    image_dir="./data/train/images",
    mask_dir="./data/train/masks",
    batch_size=16,
    image_size=(512, 512),
    augmentation_mode="heavy",  # "light", "medium", "heavy", "none"
    num_workers=8,
    pin_memory=True,
    shuffle=True
)
```

### Data Augmentation Strategies

The dataset includes several augmentation modes:

#### Light Augmentation
- Horizontal flipping (50% probability)
- Best for: Small datasets, quick experimentation

#### Medium Augmentation (Default)
- Horizontal flipping (50% probability)
- Vertical flipping (30% probability)
- Random rotation (±15 degrees)
- Color jittering on images only
- Best for: Most use cases

#### Heavy Augmentation
- All medium augmentations
- Stronger rotation (±30 degrees)
- Random cropping
- Best for: Large datasets, challenging tasks

#### Custom Augmentations
```python
from utils.data_utils import (
    SegmentationDataset, 
    DualRandomHorizontalFlip, 
    DualRandomRotation,
    ColorJitter
)

# Create custom augmentation pipeline
custom_augmentations = [
    DualRandomHorizontalFlip(p=0.7),
    DualRandomRotation(degrees=20),
    # Add more as needed
]

dataset = SegmentationDataset(
    image_dir="./data/images",
    mask_dir="./data/masks",
    augmentations=custom_augmentations,
    color_jitter=True
)
```

### Training with Custom Data

#### Basic Training
```bash
# Train with your dataset
python train.py \
    --image-dir ./data/images \
    --mask-dir ./data/masks \
    --epochs 100 \
    --batch-size 8 \
    --augmentation-mode medium
```

#### With Validation
```bash
# Train with automatic validation split
python train.py \
    --image-dir ./data/images \
    --mask-dir ./data/masks \
    --epochs 100 \
    --batch-size 8 \
    --val-split 0.2 \
    --augmentation-mode medium \
    --wandb  # Optional: for experiment tracking
```

#### Multi-size Training
```bash
# Train with larger images (requires more GPU memory)
python train.py \
    --image-dir ./data/images \
    --mask-dir ./data/masks \
    --image-size 512 \
    --batch-size 4 \
    --epochs 100
```

### Common Data Issues & Solutions

#### Issue: Mismatched Image/Mask Count
```bash
# Diagnosis
python prepare_data.py --action validate --image-dir ./data/images --mask-dir ./data/masks

# Solution: Check naming conventions and file extensions
```

#### Issue: Different Image Sizes
The dataset automatically resizes images, but for better results:
- Analyze your data: `python prepare_data.py --action analyze`
- Choose image_size based on your data's common size
- Consider aspect ratio preservation

#### Issue: Mask Values Not Binary
```python
# The dataset automatically handles thresholding
dataset = SegmentationDataset(
    image_dir="./data/images", 
    mask_dir="./data/masks",
    mask_threshold=0.5  # Adjust as needed
)
```

#### Issue: Memory Errors
- Reduce batch_size
- Reduce image_size
- Reduce num_workers
- Use pin_memory=False

### Dataset Recommendations

#### For Small Datasets (< 100 samples)
- Use "light" augmentation
- Larger learning rate
- More epochs
- Consider synthetic data mixing

#### For Medium Datasets (100-1000 samples)
- Use "medium" augmentation
- Standard parameters
- Validation split recommended

#### For Large Datasets (> 1000 samples)
- Use "heavy" augmentation
- Consider larger image sizes
- Multi-GPU training

### File Naming Conventions

The dataset automatically handles these naming patterns:

**Supported mask naming:**
- `image.jpg` → `image.png`
- `image.jpg` → `image_mask.png`
- `image.jpg` → `image_gt.png`
- `image.jpg` → `image.jpg` (same extension)

**Example valid pairs:**
```
photo_001.jpg ↔ photo_001.png
IMG_123.png   ↔ IMG_123_mask.png
data_05.tiff  ↔ data_05_gt.png
```

## Customization

### Model Parameters
Modify the model architecture in `src/models/diffusion_model.py`:
- `in_channels`: Number of input image channels (default: 3 for RGB)
- `num_classes`: Number of segmentation classes (default: 1 for binary)
- `timesteps`: Number of diffusion timesteps (default: 1000)

### Training Parameters
- Learning rate, batch size, and other hyperparameters can be adjusted via command line
- Loss function can be modified in the training script
- Data augmentation can be added to the dataset class

## Visualization

The package includes comprehensive visualization tools:
- Training loss curves
- Segmentation results comparison
- Forward diffusion process visualization
- Reverse diffusion process step-by-step
- Interactive plotting with matplotlib

## Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- Pillow >= 9.0.0
- tqdm >= 4.64.0
- wandb >= 0.13.0 (optional, for experiment tracking)

## License

[Add your license information here]

## Citation

[Add citation information if this is based on published research]

## Contributing

[Add contribution guidelines if applicable]
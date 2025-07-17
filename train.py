#!/usr/bin/env python3
"""
Training script for Diffusion Segmentation model
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm
import wandb
from typing import Optional

from src.models import DiffusionSegmentation, MorphologicalLoss
from utils.data_utils import load_dataset, create_synthetic_data
from utils.visualization import plot_training_curves, visualize_segmentation


def train_step(model: DiffusionSegmentation, image: torch.Tensor, mask: torch.Tensor, 
               optimizer: torch.optim.Optimizer, device: torch.device, 
               loss_fn=None) -> float:
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    image, mask = image.to(device), mask.to(device)
    predicted, target = model(image, mask)
    
    if loss_fn is not None:
        loss = loss_fn(predicted, target)
    else:
        loss = F.mse_loss(predicted, target)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def validate(model: DiffusionSegmentation, val_loader: DataLoader, device: torch.device, 
             loss_fn=None) -> float:
    """Validation loop"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for image, mask in val_loader:
            image, mask = image.to(device), mask.to(device)
            
            # For validation, we want to compute the training loss
            # So we need to call the model in training mode temporarily
            model.train()
            predicted, target = model(image, mask)
            model.eval()
            
            if loss_fn is not None:
                loss = loss_fn(predicted, target)
            else:
                loss = F.mse_loss(predicted, target)
                
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def save_checkpoint(model: DiffusionSegmentation, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, path: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model: DiffusionSegmentation, optimizer: torch.optim.Optimizer, 
                   path: str, device: torch.device):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Segmentation Model')
    parser.add_argument('--data-dir', type=str, help='Path to dataset directory')
    parser.add_argument('--image-dir', type=str, help='Path to images directory')
    parser.add_argument('--mask-dir', type=str, help='Path to masks directory')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=256, help='Image size')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data for testing')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--save-freq', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--val-split', type=float, default=0.0, help='Validation split ratio (0.0 = no validation)')
    parser.add_argument('--augmentation-mode', type=str, default='medium', 
                       choices=['light', 'medium', 'heavy', 'none'], help='Augmentation intensity')
    parser.add_argument('--unet-type', type=str, default='custom',
                       choices=['custom', 'diffusers_2d', 'diffusers_2d_cond'], 
                       help='Type of UNet to use')
    parser.add_argument('--pretrained-model', type=str, help='Path or name of pretrained diffusers model')
    parser.add_argument('--diffusion-type', type=str, default='gaussian',
                       choices=['gaussian', 'morphological'], 
                       help='Type of diffusion process')
    parser.add_argument('--morph-type', type=str, default='dilation',
                       choices=['dilation', 'erosion', 'mixed'], 
                       help='Type of morphological operation')
    parser.add_argument('--morph-kernel-size', type=int, default=3,
                       help='Size of morphological kernel')
    parser.add_argument('--morph-schedule', type=str, default='linear',
                       choices=['linear', 'cosine', 'quadratic'],
                       help='Schedule type for morphological intensity')
    parser.add_argument('--use-morph-loss', action='store_true',
                       help='Use morphological loss instead of MSE')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize wandb
    if args.wandb:
        wandb.init(project="diffusion-segmentation", config=config)
    
    # Create model
    model = DiffusionSegmentation(
        in_channels=3, 
        num_classes=1, 
        timesteps=args.timesteps,
        unet_type=args.unet_type,
        pretrained_model_name_or_path=args.pretrained_model,
        diffusion_type=args.diffusion_type,
        morph_type=args.morph_type,
        morph_kernel_size=args.morph_kernel_size,
        morph_schedule_type=args.morph_schedule
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Setup loss function
    loss_fn = None
    if args.use_morph_loss and args.diffusion_type == "morphological":
        loss_fn = MorphologicalLoss().to(device)
        print("Using morphological loss function")
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
        print(f"Resumed from epoch {start_epoch}")
    
    # Setup data
    if args.synthetic:
        print("Using synthetic data for training")
        train_loader = None
        val_loader = None
    else:
        if not args.image_dir or not args.mask_dir:
            raise ValueError("Must provide --image-dir and --mask-dir when not using synthetic data")
        
        # Load dataset with optional validation split
        dataset_result = load_dataset(
            args.image_dir, args.mask_dir, 
            batch_size=args.batch_size,
            image_size=(args.image_size, args.image_size),
            augmentation_mode=args.augmentation_mode,
            val_split=args.val_split
        )
        
        if args.val_split > 0:
            train_loader, val_loader = dataset_result
            print(f"Dataset split: {len(train_loader.dataset)} train, {len(val_loader.dataset)} validation")
        else:
            train_loader = dataset_result
            val_loader = None
            print(f"Training dataset: {len(train_loader.dataset)} samples")
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []
        
        # Training
        if args.synthetic:
            # Use synthetic data
            for batch_idx in tqdm(range(100), desc=f"Epoch {epoch+1}/{args.epochs}"):
                image, mask = create_synthetic_data(args.batch_size, (args.image_size, args.image_size))
                loss = train_step(model, image, mask, optimizer, device, loss_fn)
                epoch_losses.append(loss)
        else:
            # Use real data
            for batch_idx, (image, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
                loss = train_step(model, image, mask, optimizer, device, loss_fn)
                epoch_losses.append(loss)
        
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        if val_loader:
            avg_val_loss = validate(model, val_loader, device, loss_fn)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.6f}, LR: {current_lr:.2e}")
        
        # Update learning rate
        scheduler.step()
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.6f}, LR: {current_lr:.2e}")
        
        if args.wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "learning_rate": current_lr
            }
            if val_loader:
                log_dict["val_loss"] = avg_val_loss
            wandb.log(log_dict)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch + 1, avg_train_loss, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Generate sample predictions
        if (epoch + 1) % (args.save_freq * 2) == 0:
            model.eval()
            with torch.no_grad():
                if args.synthetic:
                    sample_image, sample_mask = create_synthetic_data(1, (args.image_size, args.image_size))
                else:
                    sample_image, sample_mask = next(iter(train_loader))
                    sample_image, sample_mask = sample_image[:1], sample_mask[:1]
                
                sample_image = sample_image.to(device)
                predicted_mask = model(sample_image)
                
                # Save visualization
                save_path = output_dir / f"sample_epoch_{epoch+1}.png"
                visualize_segmentation(
                    sample_image.cpu(), sample_mask, predicted_mask.cpu(),
                    title=f"Epoch {epoch+1} Results", save_path=str(save_path)
                )
    
    # Save final model
    final_model_path = output_dir / "final_model.pth"
    save_checkpoint(model, optimizer, args.epochs, train_losses[-1], final_model_path)
    print(f"Saved final model: {final_model_path}")
    
    # Plot training curves
    if val_losses:
        plot_training_curves(train_losses, val_losses, "Training vs Validation Loss", str(output_dir / "training_curves.png"))
    else:
        plot_training_curves(train_losses, "Training Loss", str(output_dir / "training_curves.png"))
    
    if args.wandb:
        wandb.finish()
    
    print("Training completed!")


if __name__ == "__main__":
    main()

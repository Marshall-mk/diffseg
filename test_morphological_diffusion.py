#!/usr/bin/env python3
"""
Test script to verify morphological diffusion functionality
"""

import torch
import torch.nn.functional as F
import numpy as np
from src.models import DiffusionSegmentation, MorphologicalLoss
import matplotlib.pyplot as plt

def create_test_mask(batch_size: int, size: int = 64) -> torch.Tensor:
    """Create test masks with simple geometric shapes."""
    masks = torch.zeros(batch_size, 1, size, size)
    
    for i in range(batch_size):
        # Create different shapes for each batch item
        center_x, center_y = size // 2, size // 2
        
        if i % 3 == 0:
            # Circle
            radius = size // 4
            y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
            circle_mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
            masks[i, 0] = circle_mask.float()
        elif i % 3 == 1:
            # Square
            square_size = size // 3
            masks[i, 0, center_y-square_size//2:center_y+square_size//2, 
                  center_x-square_size//2:center_x+square_size//2] = 1.0
        else:
            # Cross shape
            thickness = 4
            masks[i, 0, center_y-thickness:center_y+thickness, :] = 1.0
            masks[i, 0, :, center_x-thickness:center_x+thickness] = 1.0
    
    return masks

def test_morphological_diffusion():
    """Test morphological diffusion with different settings."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Test parameters
    batch_size = 4
    image_size = 64
    timesteps = 100
    
    # Create test data
    images = torch.randn(batch_size, 3, image_size, image_size).to(device)
    masks = create_test_mask(batch_size, image_size).to(device)
    
    print(f"Test data shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Masks: {masks.shape}")
    
    # Test different morphological types
    morph_types = ['dilation', 'erosion', 'mixed']
    
    for morph_type in morph_types:
        print(f"\n--- Testing Morphological Diffusion: {morph_type} ---")
        
        try:
            # Create model
            model = DiffusionSegmentation(
                in_channels=3,
                num_classes=1,
                timesteps=timesteps,
                unet_type='custom',
                diffusion_type='morphological',
                morph_type=morph_type,
                morph_kernel_size=3,
                morph_schedule_type='linear'
            ).to(device)
            
            # Test training mode
            model.train()
            predicted_mask, target_mask = model(images, masks)
            print(f"✓ Training forward pass successful")
            print(f"  Predicted shape: {predicted_mask.shape}")
            print(f"  Target shape: {target_mask.shape}")
            
            # Test loss computation
            loss_fn = MorphologicalLoss().to(device)
            loss = loss_fn(predicted_mask, target_mask)
            print(f"  Morphological loss: {loss.item():.4f}")
            
            # Test MSE loss for comparison
            mse_loss = F.mse_loss(predicted_mask, target_mask)
            print(f"  MSE loss: {mse_loss.item():.4f}")
            
            # Test inference mode
            model.eval()
            with torch.no_grad():
                output_mask = model(images)
            print(f"✓ Inference successful")
            print(f"  Output shape: {output_mask.shape}")
            print(f"  Output range: [{output_mask.min():.3f}, {output_mask.max():.3f}]")
            
            # Visualize one example
            if morph_type == 'dilation':  # Save visualization for first test
                save_visualization(images[0], masks[0], output_mask[0], f"test_{morph_type}")
            
        except Exception as e:
            print(f"✗ Error with {morph_type}: {e}")
            import traceback
            traceback.print_exc()

def save_visualization(image, original_mask, predicted_mask, filename):
    """Save a visualization of the test results."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    img_np = image.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize
    axes[0].imshow(img_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Original mask
    axes[1].imshow(original_mask.cpu().numpy().squeeze(), cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Predicted mask
    axes[2].imshow(predicted_mask.cpu().numpy().squeeze(), cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{filename}_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved as {filename}_visualization.png")

def compare_gaussian_vs_morphological():
    """Compare Gaussian and morphological diffusion side by side."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Comparing Gaussian vs Morphological Diffusion ---")
    
    # Test parameters
    batch_size = 2
    image_size = 64
    timesteps = 50
    
    # Create test data
    images = torch.randn(batch_size, 3, image_size, image_size).to(device)
    masks = create_test_mask(batch_size, image_size).to(device)
    
    diffusion_types = ['gaussian', 'morphological']
    
    for diff_type in diffusion_types:
        print(f"\n  Testing {diff_type} diffusion:")
        
        model_params = {
            'in_channels': 3,
            'num_classes': 1,
            'timesteps': timesteps,
            'unet_type': 'custom',
            'diffusion_type': diff_type
        }
        
        if diff_type == 'morphological':
            model_params.update({
                'morph_type': 'dilation',
                'morph_kernel_size': 3,
                'morph_schedule_type': 'linear'
            })
        
        model = DiffusionSegmentation(**model_params).to(device)
        
        # Training
        model.train()
        predicted, target = model(images, masks)
        
        if diff_type == 'gaussian':
            loss = F.mse_loss(predicted, target)
        else:
            loss_fn = MorphologicalLoss().to(device)
            loss = loss_fn(predicted, target)
        
        print(f"    Training loss: {loss.item():.4f}")
        
        # Inference
        model.eval()
        with torch.no_grad():
            output = model(images)
        
        print(f"    Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"    Parameter count: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    print("=== Testing Morphological Diffusion Implementation ===")
    
    test_morphological_diffusion()
    compare_gaussian_vs_morphological()
    
    print("\n=== All tests completed ===")
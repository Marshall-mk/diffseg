#!/usr/bin/env python3
"""
Test script to verify UNet switching functionality
"""

import torch
from src.models import DiffusionSegmentation

def test_unet_types():
    """Test different UNet types"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Test data
    batch_size = 2
    image_size = 64  # Small size for testing
    image = torch.randn(batch_size, 3, image_size, image_size).to(device)
    mask = torch.randn(batch_size, 1, image_size, image_size).to(device)
    
    unet_types = ['custom', 'diffusers_2d', 'diffusers_2d_cond']
    
    for unet_type in unet_types:
        print(f"\n--- Testing {unet_type} ---")
        
        try:
            # Create model
            model = DiffusionSegmentation(
                in_channels=3,
                num_classes=1,
                timesteps=100,  # Fewer timesteps for testing
                unet_type=unet_type
            ).to(device)
            
            # Test forward pass in training mode
            model.train()
            predicted_noise, actual_noise = model(image, mask)
            print(f"✓ Training forward pass successful")
            print(f"  Input shape: {image.shape}")
            print(f"  Predicted noise shape: {predicted_noise.shape}")
            print(f"  Actual noise shape: {actual_noise.shape}")
            
            # Test forward pass in inference mode
            model.eval()
            with torch.no_grad():
                output_mask = model(image)
            print(f"✓ Inference forward pass successful")
            print(f"  Output mask shape: {output_mask.shape}")
            
            # Print model parameter count
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Total parameters: {param_count:,}")
            
        except Exception as e:
            print(f"✗ Error with {unet_type}: {e}")
    
    print("\n--- Testing completed ---")

if __name__ == "__main__":
    test_unet_types()
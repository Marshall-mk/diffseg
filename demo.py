#!/usr/bin/env python3
"""
Demo script for Diffusion Segmentation model with synthetic data
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.models import DiffusionSegmentation
from utils.data_utils import create_synthetic_data
from utils.visualization import (
    visualize_segmentation, 
    visualize_diffusion_process, 
    plot_inference_steps,
    plot_training_curves
)


def demo_model_architecture():
    """Demonstrate model architecture and basic functionality"""
    print("=== Model Architecture Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = DiffusionSegmentation(in_channels=3, num_classes=1, timesteps=1000).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test forward pass
    batch_size = 2
    image_size = 256
    
    image = torch.randn(batch_size, 3, image_size, image_size).to(device)
    mask = torch.randint(0, 2, (batch_size, 1, image_size, image_size)).float().to(device)
    
    print(f"\nInput shapes:")
    print(f"Image: {image.shape}")
    print(f"Mask: {mask.shape}")
    
    # Training mode
    model.train()
    predicted_noise, actual_noise = model(image, mask)
    print(f"\nTraining mode output:")
    print(f"Predicted noise: {predicted_noise.shape}")
    print(f"Actual noise: {actual_noise.shape}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        predicted_mask = model(image)
    print(f"\nInference mode output:")
    print(f"Predicted mask: {predicted_mask.shape}")
    print(f"Mask range: [{predicted_mask.min():.3f}, {predicted_mask.max():.3f}]")


def demo_training_step():
    """Demonstrate a few training steps"""
    print("\n=== Training Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and optimizer
    model = DiffusionSegmentation(in_channels=3, num_classes=1, timesteps=1000).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    losses = []
    num_steps = 20
    
    print(f"Running {num_steps} training steps with synthetic data...")
    
    for step in range(num_steps):
        # Generate synthetic data
        image, mask = create_synthetic_data(batch_size=4, image_size=(256, 256))
        image, mask = image.to(device), mask.to(device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        predicted_noise, actual_noise = model(image, mask)
        loss = torch.nn.functional.mse_loss(predicted_noise, actual_noise)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 5 == 0:
            print(f"Step {step + 1}/{num_steps} - Loss: {loss.item():.6f}")
    
    # Plot training curve
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title("Training Loss (Demo)")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model


def demo_inference(model):
    """Demonstrate inference process"""
    print("\n=== Inference Demo ===")
    
    device = next(model.parameters()).device
    
    # Generate test data
    image, ground_truth_mask = create_synthetic_data(batch_size=1, image_size=(256, 256))
    image = image.to(device)
    
    print("Generating segmentation prediction...")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        predicted_mask = model(image)
    
    # Visualize results
    visualize_segmentation(
        image.cpu(), ground_truth_mask, predicted_mask.cpu(),
        title="Inference Results (Synthetic Data)"
    )
    
    return image, predicted_mask


def demo_diffusion_process(model):
    """Demonstrate the diffusion process"""
    print("\n=== Diffusion Process Demo ===")
    
    device = next(model.parameters()).device
    
    # Generate test image
    image, _ = create_synthetic_data(batch_size=1, image_size=(256, 256))
    image = image.to(device)
    
    print("Visualizing forward diffusion process...")
    visualize_diffusion_process(model, image, timesteps=[0, 200, 500, 800, 999])
    
    print("Visualizing reverse diffusion process (inference steps)...")
    plot_inference_steps(model, image, num_steps=8)


def demo_different_image_sizes():
    """Demonstrate model with different image sizes"""
    print("\n=== Multi-Scale Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionSegmentation(in_channels=3, num_classes=1, timesteps=1000).to(device)
    
    image_sizes = [128, 256, 512]
    
    for size in image_sizes:
        print(f"\nTesting with image size: {size}x{size}")
        
        image, mask = create_synthetic_data(batch_size=1, image_size=(size, size))
        image, mask = image.to(device), mask.to(device)
        
        # Test training forward pass
        model.train()
        predicted_noise, actual_noise = model(image, mask)
        print(f"Training - Input: {image.shape}, Output: {predicted_noise.shape}")
        
        # Test inference
        model.eval()
        with torch.no_grad():
            predicted_mask = model(image)
        print(f"Inference - Input: {image.shape}, Output: {predicted_mask.shape}")


def main():
    """Run all demos"""
    print("🚀 Diffusion Segmentation Model Demo")
    print("=" * 50)
    
    # Create output directory for saving figures
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Demo 1: Model architecture
        demo_model_architecture()
        
        # Demo 2: Training
        trained_model = demo_training_step()
        
        # Demo 3: Inference
        image, predicted_mask = demo_inference(trained_model)
        
        # Demo 4: Diffusion process
        demo_diffusion_process(trained_model)
        
        # Demo 5: Multi-scale
        demo_different_image_sizes()
        
        print("\n✅ Demo completed successfully!")
        print(f"Check the '{output_dir}' directory for saved visualizations.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
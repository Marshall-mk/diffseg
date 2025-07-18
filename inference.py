#!/usr/bin/env python3
"""
Inference script for Diffusion Segmentation model
"""

import torch
import torch.nn.functional as F
import argparse
import os
from pathlib import Path
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models import DiffusionSegmentation
from utils.data_utils import preprocess_image
from utils.visualization import visualize_segmentation, plot_inference_steps


def load_model(checkpoint_path: str, device: torch.device, timesteps: int = 1000, 
               unet_type: str = "diffusers_2d", pretrained_model_name_or_path: str = None,
               diffusion_type: str = "gaussian", morph_type: str = "dilation",
               morph_kernel_size: int = 3, morph_schedule_type: str = "linear",
               scheduler_type: str = "ddpm") -> DiffusionSegmentation:
    """Load trained model from checkpoint"""
    model = DiffusionSegmentation(
        in_channels=3, 
        num_classes=1, 
        timesteps=timesteps,
        unet_type=unet_type,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        diffusion_type=diffusion_type,
        morph_type=morph_type,
        morph_kernel_size=morph_kernel_size,
        morph_schedule_type=morph_schedule_type,
        scheduler_type=scheduler_type
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}, trained for {checkpoint.get('epoch', 'unknown')} epochs")
    return model


def predict_single_image(model: DiffusionSegmentation, image_path: str, 
                        device: torch.device, num_inference_steps: int = 50,
                        image_size: tuple = (256, 256)) -> torch.Tensor:
    """Predict segmentation mask for a single image"""
    
    # Load and preprocess image
    image = preprocess_image(image_path, image_size).to(device)
    
    # Generate prediction
    with torch.no_grad():
        # Call the sample method directly to control num_inference_steps
        predicted_mask = model.sample(image, mask=None, num_inference_steps=num_inference_steps)
    
    return predicted_mask


def batch_inference(model: DiffusionSegmentation, input_dir: str, output_dir: str,
                   device: torch.device, num_inference_steps: int = 50,
                   image_size: tuple = (256, 256), save_visualizations: bool = True):
    """Run inference on all images in a directory"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Predict mask
            predicted_mask = predict_single_image(
                model, str(image_file), device, num_inference_steps, image_size
            )
            
            # Save mask as image
            mask_np = predicted_mask.squeeze().cpu().numpy()
            mask_np = (mask_np * 255).astype(np.uint8)
            mask_image = Image.fromarray(mask_np, mode='L')
            
            output_file = output_path / f"{image_file.stem}_mask.png"
            mask_image.save(output_file)
            
            # Save visualization if requested
            if save_visualizations:
                image = preprocess_image(str(image_file), image_size)
                vis_file = output_path / f"{image_file.stem}_visualization.png"
                visualize_segmentation(
                    image, predicted_mask.cpu(), None,  # Swap order
                    title=f"Segmentation: {image_file.name}",
                    save_path=str(vis_file)
                )

        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    print(f"Inference completed. Results saved to {output_dir}")


def interactive_inference(model: DiffusionSegmentation, device: torch.device,
                         num_inference_steps: int = 50, image_size: tuple = (256, 256)):
    """Interactive inference mode"""
    
    print("Interactive Inference Mode")
    print("Enter image paths (or 'quit' to exit):")
    
    while True:
        image_path = input("\nImage path: ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            break
        
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue
        
        try:
            # Predict mask
            print("Generating segmentation...")
            predicted_mask = predict_single_image(
                model, image_path, device, num_inference_steps, image_size
            )
            
            # Load original image for visualization
            image = preprocess_image(image_path, image_size)
            
            # Show results
            visualize_segmentation(
                image, predicted_mask.cpu(), None,  # Swap order
                title=f"Segmentation: {os.path.basename(image_path)}"
            )
            
            # Ask if user wants to save
            save = input("Save result? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                output_path = input("Output path (default: mask.png): ").strip()
                if not output_path:
                    output_path = "mask.png"
                
                mask_np = predicted_mask.squeeze().cpu().numpy()
                mask_np = (mask_np * 255).astype(np.uint8)
                mask_image = Image.fromarray(mask_np, mode='L')
                mask_image.save(output_path)
                print(f"Saved to {output_path}")
        
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run inference with Diffusion Segmentation Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, help='Input image path or directory')
    parser.add_argument('--output', type=str, default='./inference_results', help='Output directory')
    parser.add_argument('--batch', action='store_true', help='Batch inference mode (process directory)')
    parser.add_argument('--interactive', action='store_true', help='Interactive inference mode')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--image-size', type=int, default=256, help='Image size for processing')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--visualize-process', action='store_true', help='Visualize inference process')
    parser.add_argument('--no-vis', action='store_true', help='Skip saving visualizations in batch mode')
    parser.add_argument('--unet-type', type=str, default='diffusers_2d',
                       choices=['diffusers_2d', 'diffusers_2d_cond'], 
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
    parser.add_argument('--scheduler-type', type=str, default='ddpm',
                       choices=['ddpm', 'ddim'],
                       help='Type of diffusers scheduler to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device, args.timesteps, args.unet_type, args.pretrained_model,
                      args.diffusion_type, args.morph_type, args.morph_kernel_size, args.morph_schedule, args.scheduler_type)
    
    image_size = (args.image_size, args.image_size)
    
    if args.interactive:
        # Interactive mode
        interactive_inference(model, device, args.steps, image_size)
    
    elif args.batch:
        # Batch inference mode
        if not args.input:
            print("Error: --input directory required for batch mode")
            return
        
        batch_inference(
            model, args.input, args.output, device, 
            args.steps, image_size, not args.no_vis
        )
    
    else:
        # Single image inference
        if not args.input:
            print("Error: --input image path required")
            return
        
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return
        
        print(f"Processing {args.input}...")
        
        # Predict mask
        predicted_mask = predict_single_image(model, args.input, device, args.steps, image_size)
        
        # Load original image
        image = preprocess_image(args.input, image_size)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save mask
        mask_np = predicted_mask.squeeze().cpu().numpy()
        mask_np = (mask_np * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_np, mode='L')
        
        input_name = Path(args.input).stem
        mask_file = output_path / f"{input_name}_mask.png"
        mask_image.save(mask_file)
        print(f"Saved mask: {mask_file}")
        
        # Save visualization
        vis_file = output_path / f"{input_name}_visualization.png"
        visualize_segmentation(
            image, predicted_mask.cpu(), None,  # Swap order: image, predicted, ground_truth
            title=f"Segmentation: {Path(args.input).name}",
            save_path=str(vis_file)
        )
        plt.close()
        
        print(f"Saved visualization: {vis_file}")
        
        # Visualize inference process if requested
        if args.visualize_process:
            process_file = output_path / f"{input_name}_process.png"
            plot_inference_steps(
                model, image.to(device), num_steps=10, 
                save_path=str(process_file)
            )
            print(f"Saved process visualization: {process_file}")


if __name__ == "__main__":
    main()

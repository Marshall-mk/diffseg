import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Union

try:
    from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

from .morphological_ops import SoftMorphology, create_morph_schedule, apply_morphological_degradation


class DiffusionSegmentation(nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 num_classes: int = 1, 
                 timesteps: int = 1000,
                 unet_type: str = "diffusers_2d",
                 pretrained_model_name_or_path: Optional[str] = None,
                 diffusion_type: str = "gaussian",
                 morph_type: str = "dilation",
                 morph_kernel_size: int = 3,
                 morph_schedule_type: str = "linear",
                 scheduler_type: str = "ddpm"):
        """
        Initialize DiffusionSegmentation model.
        
        Args:
            in_channels: Number of input image channels (default: 3)
            num_classes: Number of segmentation classes (default: 1) 
            timesteps: Number of diffusion timesteps (default: 1000)
            unet_type: Type of UNet to use. Options: "diffusers_2d", "diffusers_2d_cond"
            pretrained_model_name_or_path: Path or name of pretrained diffusers model
            diffusion_type: Type of diffusion process. Options: "gaussian", "morphological"
            morph_type: Type of morphological operation. Options: "dilation", "erosion", "mixed"
            morph_kernel_size: Size of morphological kernel (default: 3)
            morph_schedule_type: Schedule type for morphological intensity. Options: "linear", "cosine", "quadratic"
            scheduler_type: Type of diffusers scheduler. Options: "ddpm", "ddim"
        """
        super().__init__()
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.unet_type = unet_type
        self.diffusion_type = diffusion_type
        self.morph_type = morph_type
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library not available. Install with: pip install diffusers")
        
        # Initialize UNet based on type
        if unet_type == "diffusers_2d":
            if pretrained_model_name_or_path:
                self.unet = UNet2DModel.from_pretrained(pretrained_model_name_or_path)
            else:
                self.unet = UNet2DModel(
                    sample_size=None,  # Will be inferred from input
                    in_channels=in_channels + num_classes,
                    out_channels=num_classes,
                    layers_per_block=2,
                    block_out_channels=(64, 128, 256, 512),
                    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
                )
        elif unet_type == "diffusers_2d_cond":
            if pretrained_model_name_or_path:
                self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path)
            else:
                self.unet = UNet2DConditionModel(
                    sample_size=None,  # Will be inferred from input
                    in_channels=in_channels + num_classes,
                    out_channels=num_classes,
                    layers_per_block=2,
                    block_out_channels=(64, 128, 256, 512),
                    down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
                    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
                    cross_attention_dim=768,  # Standard dimension for conditioning
                )
        else:
            raise ValueError(f"Unsupported unet_type: {unet_type}. Choose from: 'diffusers_2d', 'diffusers_2d_cond'")
        
        # Initialize diffusion schedules based on type
        if diffusion_type == "gaussian":
            # Use diffusers scheduler instead of custom implementation
            if scheduler_type == "ddpm":
                self.scheduler = DDPMScheduler(
                    num_train_timesteps=timesteps,
                    beta_schedule="scaled_linear",
                    beta_start=0.00085,
                    beta_end=0.012,
                    clip_sample=False,
                )
            elif scheduler_type == "ddim":
                self.scheduler = DDIMScheduler(
                    num_train_timesteps=timesteps,
                    beta_schedule="scaled_linear",
                    beta_start=0.00085,
                    beta_end=0.012,
                    clip_sample=False,
                )
            else:
                raise ValueError(f"Unsupported scheduler_type: {scheduler_type}. Choose from: 'ddpm', 'ddim'")
        elif diffusion_type == "morphological":
            # Morphological operations
            self.soft_morph = SoftMorphology(kernel_size=morph_kernel_size, temperature=0.5)
            self.register_buffer('morph_schedule', create_morph_schedule(timesteps, morph_schedule_type))
        else:
            raise ValueError(f"Unsupported diffusion_type: {diffusion_type}. Choose from: 'gaussian', 'morphological'")

    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process using diffusers scheduler"""
        noise = torch.randn_like(x0)
        noisy_x = self.scheduler.add_noise(x0, noise, t)
        return noisy_x, noise

    def forward_morphology(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply forward morphological process."""
        batch_size = x0.shape[0]
        device = x0.device
        
        # Get morphological intensity for this timestep
        morph_intensity = self.morph_schedule[t]
        
        result = torch.zeros_like(x0)
        
        for i in range(batch_size):
            intensity = morph_intensity[i].item()
            current_mask = x0[i:i+1]
            
            # Apply morphological degradation
            degraded_mask = apply_morphological_degradation(
                current_mask, intensity, self.morph_type, self.soft_morph
            )
            
            result[i:i+1] = degraded_mask
        
        return result

    def _call_unet(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Call UNet with appropriate parameters based on type."""
        if self.unet_type == "diffusers_2d":
            # UNet2DModel expects (sample, timestep)
            return self.unet(x, t).sample
        elif self.unet_type == "diffusers_2d_cond":
            # UNet2DConditionModel expects (sample, timestep, encoder_hidden_states)
            # For segmentation, we don't use conditioning, so pass None
            return self.unet(x, t, encoder_hidden_states=None).sample
        else:
            raise ValueError(f"Unsupported unet_type: {self.unet_type}")

    def forward(self, image: torch.Tensor, mask: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        if self.training:
            if mask is None:
                raise ValueError("Mask is required during training")
            if t is None:
                t = torch.randint(0, self.timesteps, (image.shape[0],), device=image.device)
            
            if self.diffusion_type == "gaussian":
                # Gaussian diffusion: predict noise
                noisy_mask, noise = self.forward_diffusion(mask, t)
                
                # Concatenate image and noisy mask
                x = torch.cat([image, noisy_mask], dim=1)
                
                # Predict noise
                predicted_noise = self._call_unet(x, t)
                return predicted_noise, noise
                
            elif self.diffusion_type == "morphological":
                # Morphological diffusion: predict original mask
                morphed_mask = self.forward_morphology(mask, t)
                
                # Concatenate image and morphed mask
                x = torch.cat([image, morphed_mask], dim=1)
                
                # Predict original mask
                predicted_mask = self._call_unet(x, t)
                return predicted_mask, mask
                
        else:
            # Inference mode
            if self.diffusion_type == "gaussian":
                # Start from pure noise
                if mask is None:
                    mask = torch.randn(image.shape[0], self.num_classes, image.shape[2], image.shape[3], device=image.device)
            elif self.diffusion_type == "morphological":
                # Start from binary state
                if mask is None:
                    if self.morph_type == "erosion":
                        # Start from all ones for erosion
                        mask = torch.ones(image.shape[0], self.num_classes, image.shape[2], image.shape[3], device=image.device)
                    else:
                        # Start from all zeros for dilation
                        mask = torch.zeros(image.shape[0], self.num_classes, image.shape[2], image.shape[3], device=image.device)
            
            return self.sample(image, mask)

    def sample(self, image: torch.Tensor, mask: torch.Tensor = None, num_inference_steps: int = 50) -> torch.Tensor:
        # Initialize mask if not provided
        if mask is None:
            if self.diffusion_type == "gaussian":
                # Start from pure noise
                mask = torch.randn(image.shape[0], self.num_classes, image.shape[2], image.shape[3], device=image.device)
            elif self.diffusion_type == "morphological":
                # Start from binary state
                if self.morph_type == "erosion":
                    # Start from all ones for erosion
                    mask = torch.ones(image.shape[0], self.num_classes, image.shape[2], image.shape[3], device=image.device)
                else:
                    # Start from all zeros for dilation
                    mask = torch.zeros(image.shape[0], self.num_classes, image.shape[2], image.shape[3], device=image.device)
        
        if self.diffusion_type == "gaussian":
            # Use diffusers scheduler for sampling
            self.scheduler.set_timesteps(num_inference_steps)
            
            for t in self.scheduler.timesteps:
                t_tensor = torch.full((image.shape[0],), t, device=image.device, dtype=torch.long)
                
                # Predict noise
                with torch.no_grad():
                    x = torch.cat([image, mask], dim=1)
                    predicted_noise = self._call_unet(x, t_tensor)
                
                # Use scheduler to compute previous sample
                mask = self.scheduler.step(predicted_noise, t, mask).prev_sample
            
            return torch.sigmoid(mask)
            
        elif self.diffusion_type == "morphological":
            # Reverse morphological process (unchanged)
            step_size = self.timesteps // num_inference_steps
            for i in reversed(range(0, self.timesteps, step_size)):
                t = torch.full((image.shape[0],), i, device=image.device, dtype=torch.long)
                
                # Predict the clean mask
                with torch.no_grad():
                    x = torch.cat([image, mask], dim=1)
                    predicted_clean_mask = self._call_unet(x, t)
                
                # Update mask towards predicted clean mask
                alpha = 1.0 - (i / self.timesteps)  # Interpolation weight
                mask = alpha * predicted_clean_mask + (1 - alpha) * mask
                
                # Apply constraints
                mask = torch.clamp(mask, 0, 1)
            
            return torch.sigmoid(mask)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from .unet import UNet


class DiffusionSegmentation(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, timesteps: int = 1000):
        super().__init__()
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.unet = UNet(in_channels + num_classes, num_classes)
        
        # Noise schedule
        self.register_buffer('betas', self._cosine_beta_schedule(timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        x_t = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def forward(self, image: torch.Tensor, mask: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        if self.training:
            if mask is None:
                raise ValueError("Mask is required during training")
            if t is None:
                t = torch.randint(0, self.timesteps, (image.shape[0],), device=image.device)
            
            # Forward diffusion on mask
            noisy_mask, noise = self.forward_diffusion(mask, t)
            
            # Concatenate image and noisy mask
            x = torch.cat([image, noisy_mask], dim=1)
            
            # Predict noise
            predicted_noise = self.unet(x, t)
            return predicted_noise, noise
        else:
            # Inference: start from pure noise
            if mask is None:
                mask = torch.randn(image.shape[0], self.num_classes, image.shape[2], image.shape[3], device=image.device)
            
            return self.sample(image, mask)

    def sample(self, image: torch.Tensor, mask: torch.Tensor, num_inference_steps: int = 50) -> torch.Tensor:
        # DDPM sampling
        step_size = self.timesteps // num_inference_steps
        
        for i in reversed(range(0, self.timesteps, step_size)):
            t = torch.full((image.shape[0],), i, device=image.device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                x = torch.cat([image, mask], dim=1)
                predicted_noise = self.unet(x, t)
            
            # Compute denoised mask
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            # Reshape for broadcasting
            while len(alpha_t.shape) < len(mask.shape):
                alpha_t = alpha_t.unsqueeze(-1)
                alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
                beta_t = beta_t.unsqueeze(-1)
            
            mask = (1 / torch.sqrt(alpha_t)) * (mask - beta_t * predicted_noise / torch.sqrt(1 - alpha_cumprod_t))
            
            # Add noise if not the last step
            if i > 0:
                mask = mask + torch.sqrt(beta_t) * torch.randn_like(mask)
        
        return torch.sigmoid(mask)
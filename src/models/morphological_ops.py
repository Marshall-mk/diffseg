"""
Morphological operations for diffusion segmentation.
Implements differentiable morphological operations as an alternative to Gaussian noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SoftMorphology(nn.Module):
    """Differentiable morphological operations using soft approximations."""
    
    def __init__(self, kernel_size: int = 3, temperature: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.temperature = temperature
        
        # Create structural element (circular kernel)
        self.register_buffer('kernel', self._create_circular_kernel(kernel_size))
    
    def _create_circular_kernel(self, size: int) -> torch.Tensor:
        """Create a circular structural element."""
        center = size // 2
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        kernel = ((x - center)**2 + (y - center)**2) <= (center**2)
        return kernel.float()
    
    def soft_dilation(self, x: torch.Tensor) -> torch.Tensor:
        """Soft dilation using max-pooling with temperature scaling."""
        # Unfold to get neighborhoods
        unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size//2)
        # Reshape: [B, C, kernel_size^2, H*W]
        B, C = x.shape[:2]
        H, W = x.shape[2:]
        unfolded = unfolded.view(B, C, self.kernel_size**2, H, W)
        
        # Apply structural element mask
        kernel_mask = self.kernel.view(1, 1, -1, 1, 1)
        masked = unfolded * kernel_mask + (1 - kernel_mask) * (-float('inf'))
        
        # Soft maximum using temperature-scaled softmax
        weights = F.softmax(masked / self.temperature, dim=2)
        result = (unfolded * weights).sum(dim=2)
        
        return result
    
    def soft_erosion(self, x: torch.Tensor) -> torch.Tensor:
        """Soft erosion using min-pooling with temperature scaling."""
        # Unfold to get neighborhoods
        unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size//2)
        B, C = x.shape[:2]
        H, W = x.shape[2:]
        unfolded = unfolded.view(B, C, self.kernel_size**2, H, W)
        
        # Apply structural element mask
        kernel_mask = self.kernel.view(1, 1, -1, 1, 1)
        masked = unfolded * kernel_mask + (1 - kernel_mask) * float('inf')
        
        # Soft minimum using temperature-scaled softmin
        weights = F.softmax(-masked / self.temperature, dim=2)
        result = (unfolded * weights).sum(dim=2)
        
        return result
    
    def soft_opening(self, x: torch.Tensor) -> torch.Tensor:
        """Opening = erosion followed by dilation."""
        return self.soft_dilation(self.soft_erosion(x))
    
    def soft_closing(self, x: torch.Tensor) -> torch.Tensor:
        """Closing = dilation followed by erosion."""
        return self.soft_erosion(self.soft_dilation(x))


class ConvolutionalMorphology(nn.Module):
    """Differentiable morphological operations using learned convolutions."""
    
    def __init__(self, kernel_size: int = 3, operation: str = 'dilation'):
        super().__init__()
        self.operation = operation
        
        # Learnable morphological kernel
        self.morph_conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        
        # Initialize with circular pattern
        with torch.no_grad():
            center = kernel_size // 2
            y, x = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size), indexing='ij')
            circular_mask = ((x - center)**2 + (y - center)**2) <= (center**2)
            self.morph_conv.weight.data = circular_mask.float().unsqueeze(0).unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply morphological operation."""
        # Apply convolution
        conv_result = self.morph_conv(x)
        
        if self.operation == 'dilation':
            # Dilation: any neighbor > 0 makes output > 0
            return torch.sigmoid(conv_result * 10)  # Sharp transition
        elif self.operation == 'erosion':
            # Erosion: all neighbors must be > 0
            kernel_size = self.morph_conv.weight.shape[-1]
            num_ones = (self.morph_conv.weight > 0.5).sum()
            return torch.sigmoid((conv_result - num_ones + 1) * 10)
        else:
            return conv_result


class MorphologicalLoss(nn.Module):
    """Custom loss for morphological diffusion."""
    
    def __init__(self, l1_weight: float = 1.0, dice_weight: float = 1.0, boundary_weight: float = 0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice loss for segmentation."""
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice
    
    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Boundary-aware loss to preserve morphological structure."""
        # Compute gradients to detect boundaries
        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        # L1 loss on gradients
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return (loss_x + loss_y) / 2
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combined loss."""
        l1 = F.l1_loss(pred, target)
        dice = self.dice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        return (self.l1_weight * l1 + 
                self.dice_weight * dice + 
                self.boundary_weight * boundary)


def create_morph_schedule(timesteps: int, schedule_type: str = "linear") -> torch.Tensor:
    """Create schedule for morphological operation intensity."""
    if schedule_type == "linear":
        return torch.linspace(0, 1, timesteps)
    elif schedule_type == "cosine":
        x = torch.linspace(0, 1, timesteps)
        return (1 - torch.cos(x * np.pi)) / 2
    elif schedule_type == "quadratic":
        x = torch.linspace(0, 1, timesteps)
        return x ** 2
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def apply_morphological_degradation(mask: torch.Tensor, 
                                  intensity: float, 
                                  morph_type: str = "dilation",
                                  soft_morph: SoftMorphology = None) -> torch.Tensor:
    """Apply morphological degradation with given intensity."""
    if soft_morph is None:
        soft_morph = SoftMorphology(kernel_size=3, temperature=0.5)
    
    result = mask.clone()
    num_operations = max(1, int(intensity * 10))  # 1 to 10 operations
    
    for _ in range(num_operations):
        if morph_type == "dilation":
            result = soft_morph.soft_dilation(result)
        elif morph_type == "erosion":
            result = soft_morph.soft_erosion(result)
        elif morph_type == "opening":
            result = soft_morph.soft_opening(result)
        elif morph_type == "closing":
            result = soft_morph.soft_closing(result)
        elif morph_type == "mixed":
            # Randomly choose operation
            if np.random.random() > 0.5:
                result = soft_morph.soft_dilation(result)
            else:
                result = soft_morph.soft_erosion(result)
        
        # Clip to valid range
        result = torch.clamp(result, 0, 1)
    
    return result
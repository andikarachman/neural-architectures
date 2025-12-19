"""
VGG11 with Batch Normalization (VGG11-BN)

This is an enhanced version of VGG11 that includes Batch Normalization layers.
While the original VGG paper (2014) didn't use BatchNorm, adding it provides significant benefits.

BATCH NORMALIZATION EXPLAINED:
================================
Batch Normalization (Ioffe & Szegedy, 2015) normalizes layer inputs across the mini-batch:
    
    1. Calculate mean and variance across the batch dimension
    2. Normalize: x_norm = (x - mean) / sqrt(variance + epsilon)
    3. Scale and shift: y = gamma * x_norm + beta
       (gamma and beta are learnable parameters)

WHY BATCH NORMALIZATION?
========================
1. FASTER TRAINING: Allows higher learning rates (often 10x faster convergence)
2. REDUCES INTERNAL COVARIATE SHIFT: Stabilizes distribution of layer inputs during training
3. REGULARIZATION EFFECT: Adds slight noise, reducing need for dropout (though we keep it here)
4. LESS SENSITIVE TO INITIALIZATION: Network trains well even with poor weight initialization
5. IMPROVES GRADIENT FLOW: Prevents vanishing/exploding gradients in deep networks

PLACEMENT IN ARCHITECTURE:
==========================
Standard order: Conv -> BatchNorm -> ReLU
- BatchNorm comes AFTER convolution, BEFORE activation
- This normalizes the pre-activation values
- Alternative: Conv -> ReLU -> BatchNorm (less common, different behavior)

KEY DIFFERENCES FROM ORIGINAL VGG11:
====================================
- Adds BatchNorm2d after every Conv2d layer
- Typically trains faster and achieves slightly better accuracy
- More stable training with larger learning rates
- Slightly more parameters (gamma and beta for each feature map)

Architecture Pattern:
    Conv-BN-ReLU-Pool -> Conv-BN-ReLU-Pool -> Conv-BN-Conv-BN-ReLU-Pool -> 
    Conv-BN-Conv-BN-ReLU-Pool -> Conv-BN-Conv-BN-ReLU-Pool -> FC-FC-FC
"""

from typing import Any

import torch
import torch.nn as nn

class VGG11_BN(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        """
        Initialize VGG11 with Batch Normalization.
        
        Args:
            num_classes: Number of output classes (default: 1000 for ImageNet)
            dropout: Dropout probability for regularization (default: 0.5)
                    Note: With BatchNorm, you can often use lower dropout (e.g., 0.3)
        
        Input shape: (batch_size, 3, 224, 224)
        Output shape: (batch_size, num_classes)
        """
        super().__init__()

        # Input: (N, 3, 224, 224)
        # N = batch size (must be > 1 for BatchNorm during training)

        # ========== BLOCK 1: Initial Feature Extraction with BatchNorm ==========
        # Purpose: Extract low-level features with normalized activations
        # Spatial reduction: 224x224 -> 112x112
        # Channel expansion: 3 -> 64
        self.block1 = nn.Sequential(
            # Convolution: Applies 64 filters to extract features
            nn.Conv2d(3, 64, kernel_size=3, padding=1),   # (N, 64, 224, 224)
            
            # BatchNorm2d: Normalizes the 64 feature maps
            # - Maintains separate mean/variance statistics for each of the 64 channels
            # - During training: uses batch statistics (mean and variance of current batch)
            # - During inference: uses running statistics (exponential moving average from training)
            # - Learnable parameters: 64 gammas (scale) + 64 betas (shift) = 128 params
            # - Effect: Makes training more stable and allows higher learning rates
            nn.BatchNorm2d(64),
            
            # ReLU: Applied AFTER normalization
            # This ensures activations are already normalized before non-linearity
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2)         # (N, 64, 112, 112)   
        )

        # ========== BLOCK 2: Mid-level Features with BatchNorm ==========
        # Purpose: Combine low-level features with stable gradient flow
        # Spatial reduction: 112x112 -> 56x56
        # Channel expansion: 64 -> 128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (N, 128, 112, 112)
            
            # BatchNorm helps each layer receive inputs with consistent distribution
            # Without BN: Conv outputs might have very different scales/distributions
            # With BN: Each layer gets normalized inputs, making optimization easier
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)         # (N, 128, 56, 56)   
        )

        # ========== BLOCK 3: Higher-level Features with BatchNorm ==========
        # Purpose: Learn complex patterns with normalized intermediate representations
        # Spatial reduction: 56x56 -> 28x28
        # Channel expansion: 128 -> 256
        # NOTE: TWO conv layers → TWO BatchNorm layers
        self.block3 = nn.Sequential(
            # First Conv-BN-ReLU unit
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# (N, 256, 56, 56)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Second Conv-BN-ReLU unit
            # BatchNorm between convolutions is crucial:
            # - Prevents internal covariate shift (input distribution changes during training)
            # - Each conv layer receives normalized inputs from previous layer
            # - Dramatically improves gradient flow through deep networks
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# (N, 256, 56, 56)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)         # (N, 256, 28, 28)
        )

        # ========== BLOCK 4: High-level Semantic Features with BatchNorm ==========
        # Purpose: Learn object-level features with stable training dynamics
        # Spatial reduction: 28x28 -> 14x14
        # Channel expansion: 256 -> 512
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),# (N, 512, 28, 28)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # At this depth, BatchNorm becomes increasingly important:
            # - Deeper networks suffer more from vanishing/exploding gradients
            # - BatchNorm keeps activations and gradients in healthy ranges
            # - Allows the network to train effectively even at this depth
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# (N, 512, 28, 28)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)         # (N, 512, 14, 14)
        )

        # ========== BLOCK 5: Final Feature Refinement with BatchNorm ==========
        # Purpose: Refine high-level features with normalized representations
        # Spatial reduction: 14x14 -> 7x7
        # Channels: Stays at 512
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# (N, 512, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Final convolutional processing
            # BatchNorm here ensures the features fed to classifier are normalized
            # This makes the fully connected layers easier to train
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# (N, 512, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)         # (N, 512, 7, 7)
        )

        # ========== CLASSIFIER HEAD: Decision Making ==========
        # Note: We don't typically use BatchNorm in fully connected layers for classification
        # The feature extraction backbone already provides well-normalized features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),                  # (N, 4096)
            nn.ReLU(inplace=True),
            
            # Dropout: Still useful even with BatchNorm
            # - BatchNorm provides regularization during training (batch statistics add noise)
            # - Dropout provides additional regularization by forcing redundancy
            # - Many practitioners reduce dropout rate when using BatchNorm (e.g., 0.3 instead of 0.5)
            # - Both techniques work differently and can complement each other
            nn.Dropout(dropout),
            
            nn.Linear(4096, 4096),                         # (N, 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Final layer: No BatchNorm here
            # We want raw logits for the loss function (e.g., CrossEntropyLoss)
            nn.Linear(4096, num_classes)                   # (N, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VGG11 with Batch Normalization.
        
        IMPORTANT BEHAVIORAL DIFFERENCE BETWEEN TRAINING AND INFERENCE:
        ---------------------------------------------------------------
        Training mode (model.train()):
            - BatchNorm uses current batch statistics (mean, std)
            - Updates running statistics using exponential moving average
            - Dropout is active (randomly drops neurons)
            
        Inference mode (model.eval()):
            - BatchNorm uses accumulated running statistics from training
            - Provides deterministic output (no batch dependency)
            - Dropout is disabled (uses all neurons)
            
        This is why you must call model.eval() before inference!
        
        Args:
            x: Input tensor of shape (N, 3, 224, 224)
               N should be > 1 during training for meaningful batch statistics
            
        Returns:
            Output tensor of shape (N, num_classes) containing raw logits
        """
        # Feature extraction with BatchNorm normalization at each stage
        x = self.block1(x)      # (N, 3, 224, 224) -> (N, 64, 112, 112)
        x = self.block2(x)      # (N, 64, 112, 112) -> (N, 128, 56, 56)
        x = self.block3(x)      # (N, 128, 56, 56) -> (N, 256, 28, 28)
        x = self.block4(x)      # (N, 256, 28, 28) -> (N, 512, 14, 14)
        x = self.block5(x)      # (N, 512, 14, 14) -> (N, 512, 7, 7)

        # Flatten: Convert 3D feature maps to 1D feature vector
        x = torch.flatten(x, 1)  # (N, 512, 7, 7) -> (N, 25088)
        
        # Classification: Transform normalized features to class predictions
        x = self.classifier(x)   # (N, 25088) -> (N, num_classes)
        return x
    

if __name__ == "__main__":
    """
    Example usage demonstrating VGG11 with Batch Normalization.
    
    KEY TAKEAWAYS:
    --------------
    1. BatchNorm adds minimal parameters but significant training benefits
    2. Always call model.eval() before inference to use running statistics
    3. Batch size should be > 1 during training for meaningful batch statistics
    4. Training with BatchNorm is typically 5-10x faster than without
    """
    
    # Create model for ImageNet classification
    model = VGG11_BN(num_classes=1000)
    
    # IMPORTANT: Set to eval mode for inference
    # This switches BatchNorm to use running statistics instead of batch statistics
    model.eval()
    
    # Create sample input
    # Note: Batch size of 1 is fine for inference when model is in eval mode
    # During training, use batch_size > 1 (typically 32, 64, 128, etc.)
    sample_input = torch.randn(1, 3, 224, 224)
    
    # Forward pass (no gradient computation needed for inference)
    with torch.no_grad():
        sample_output = model(sample_input)
    
    print("=" * 60)
    print("VGG11 with Batch Normalization - Architecture Summary")
    print("=" * 60)
    print(f"Input shape:  {sample_input.shape}")   # torch.Size([1, 3, 224, 224])
    print(f"Output shape: {sample_output.shape}")  # torch.Size([1, 1000])
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count BatchNorm parameters separately
    bn_params = sum(p.numel() for name, p in model.named_parameters() 
                    if 'bn' in name.lower() or 'batch_norm' in name.lower())
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"BatchNorm parameters: {bn_params:,}")
    print(f"Non-BN parameters: {total_params - bn_params:,}")





"""
VGG11 Neural Network Architecture

VGG (Visual Geometry Group) was introduced in the paper "Very Deep Convolutional Networks 
for Large-Scale Image Recognition" by Simonyan & Zisserman (2014).

Key Design Principles:
1. Uses only 3x3 convolution kernels throughout (smallest size to capture spatial patterns)
2. Uses 2x2 max pooling with stride 2 for downsampling
3. Doubles the number of channels after each pooling layer (64 -> 128 -> 256 -> 512)
4. Deep architecture with simple, uniform structure
5. ReLU activation after every convolution

VGG11 specifically has:
- 8 convolutional layers (arranged in 5 blocks)
- 3 fully connected layers
- 11 weight layers total (hence the name VGG11)

Architecture Pattern:
    Conv-ReLU-Pool -> Conv-ReLU-Pool -> Conv-Conv-ReLU-Pool -> 
    Conv-Conv-ReLU-Pool -> Conv-Conv-ReLU-Pool -> FC-FC-FC
"""

from typing import Any

import torch
import torch.nn as nn

class VGG11(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        """
        Initialize VGG11 architecture.
        
        Args:
            num_classes: Number of output classes (default: 1000 for ImageNet)
            dropout: Dropout probability for regularization (default: 0.5)
        
        Input shape: (batch_size, 3, 224, 224)
        Output shape: (batch_size, num_classes)
        """
        super().__init__()

        # Input: (N, 3, 224, 224)
        # N = batch size, 3 = RGB channels, 224x224 = image dimensions

        # ========== BLOCK 1: Initial Feature Extraction ==========
        # Purpose: Extract low-level features (edges, colors, simple textures)
        # Spatial reduction: 224x224 -> 112x112 (halved by pooling)
        # Channel expansion: 3 -> 64 (learns 64 different feature detectors)
        self.block1 = nn.Sequential(
            # Conv2d: Applies 64 different 3x3 filters to the input image
            # - in_channels=3: RGB input
            # - out_channels=64: produces 64 feature maps
            # - kernel_size=3: uses 3x3 filters (captures local patterns)
            # - padding=1: adds 1 pixel border to maintain spatial dimensions
            nn.Conv2d(3, 64, kernel_size=3, padding=1),   # (N, 64, 224, 224)
            
            # ReLU: Introduces non-linearity, allows network to learn complex patterns
            # inplace=True saves memory by modifying the input directly
            nn.ReLU(inplace=True),
            
            # MaxPool2d: Downsamples by taking maximum value in each 2x2 window
            # - Reduces spatial dimensions by half (224 -> 112)
            # - Provides translation invariance (small shifts don't affect output much)
            # - Reduces computational cost for subsequent layers
            nn.MaxPool2d(kernel_size=2, stride=2)         # (N, 64, 112, 112)   
        )

        # ========== BLOCK 2: Mid-level Feature Learning ==========
        # Purpose: Combine low-level features into more complex patterns
        # Spatial reduction: 112x112 -> 56x56
        # Channel expansion: 64 -> 128 (doubles the feature capacity)
        self.block2 = nn.Sequential(
            # Each of the 128 filters operates on all 64 input channels
            # This creates combinations of the low-level features from Block 1
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (N, 128, 112, 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)         # (N, 128, 56, 56)   
        )

        # ========== BLOCK 3: Higher-level Feature Abstraction ==========
        # Purpose: Learn complex object parts and patterns
        # Spatial reduction: 56x56 -> 28x28
        # Channel expansion: 128 -> 256
        # NOTE: This block has TWO conv layers before pooling (deeper processing)
        self.block3 = nn.Sequential(
            # First conv: Expands feature channels from 128 to 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# (N, 256, 56, 56)
            nn.ReLU(inplace=True),
            
            # Second conv: Processes the 256 features at the same spatial resolution
            # Two stacked 3x3 convs have effective receptive field of 5x5
            # This is more efficient than using a single 5x5 conv (fewer parameters)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# (N, 256, 56, 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)         # (N, 256, 28, 28)
        )

        # ========== BLOCK 4: High-level Semantic Features ==========
        # Purpose: Learn object-level features and semantically meaningful patterns
        # Spatial reduction: 28x28 -> 14x14
        # Channel expansion: 256 -> 512 (reaches maximum channel capacity)
        # Structure: Same as Block 3 (two conv layers before pooling)
        self.block4 = nn.Sequential(
            # Expands to 512 channels - maximum capacity for rich feature representation
            nn.Conv2d(256, 512, kernel_size=3, padding=1),# (N, 512, 28, 28)
            nn.ReLU(inplace=True),
            
            # Processes features at 512 channels
            # By this point, the network has seen enough context to understand objects
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# (N, 512, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)         # (N, 512, 14, 14)
        )

        # ========== BLOCK 5: Final Feature Refinement ==========
        # Purpose: Refine high-level features for classification
        # Spatial reduction: 14x14 -> 7x7 (final spatial dimension)
        # Channels: Stays at 512 (no expansion needed)
        self.block5 = nn.Sequential(
            # Maintains 512 channels - focuses on refining existing features
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# (N, 512, 14, 14)
            nn.ReLU(inplace=True),
            
            # Final convolutional processing before classification
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# (N, 512, 14, 14)
            nn.ReLU(inplace=True),
            
            # Final pooling: Results in 7x7 spatial dimension
            # Total spatial reduction: 224 -> 112 -> 56 -> 28 -> 14 -> 7 (32x smaller)
            nn.MaxPool2d(kernel_size=2, stride=2)         # (N, 512, 7, 7)
        )

        # ========== CLASSIFIER HEAD: Decision Making ==========
        # Purpose: Transform spatial features into class predictions
        # Input: (N, 512, 7, 7) = 25,088 features per sample
        # Output: (N, num_classes) = probability distribution over classes
        self.classifier = nn.Sequential(
            # First FC layer: Compresses 25,088 features to 4096
            # This is where spatial information is fully aggregated
            # 512 * 7 * 7 = 25,088 input features
            nn.Linear(512 * 7 * 7, 4096),                  # (N, 4096)
            nn.ReLU(inplace=True),
            
            # Dropout: Randomly zeros 50% of neurons during training
            # - Prevents overfitting by forcing network to learn redundant representations
            # - During inference (eval mode), dropout is disabled
            nn.Dropout(dropout),
            
            # Second FC layer: Maintains 4096 dimensions
            # Learns complex combinations of the compressed features
            nn.Linear(4096, 4096),                         # (N, 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Final FC layer: Maps to number of output classes
            # No activation here - raw logits are output
            # (Softmax is typically applied in the loss function for numerical stability)
            nn.Linear(4096, num_classes)                   # (N, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VGG11.
        
        The data flows through:
        1. Feature extraction blocks (conv + pooling) - learn hierarchical features
        2. Flatten operation - convert 3D features to 1D vector
        3. Classifier layers (fully connected) - make final predictions
        
        Args:
            x: Input tensor of shape (N, 3, 224, 224)
            
        Returns:
            Output tensor of shape (N, num_classes) containing raw logits
        """
        # Feature extraction: Each block progressively learns higher-level features
        # Block 1: edges, colors, simple textures
        x = self.block1(x)      # (N, 3, 224, 224) -> (N, 64, 112, 112)
        
        # Block 2: corners, simple shapes
        x = self.block2(x)      # (N, 64, 112, 112) -> (N, 128, 56, 56)
        
        # Block 3: object parts, patterns
        x = self.block3(x)      # (N, 128, 56, 56) -> (N, 256, 28, 28)
        
        # Block 4: object components, semantic elements
        x = self.block4(x)      # (N, 256, 28, 28) -> (N, 512, 14, 14)
        
        # Block 5: complete objects, high-level concepts
        x = self.block5(x)      # (N, 512, 14, 14) -> (N, 512, 7, 7)

        # Flatten: Convert spatial features to vector
        # - Dimension 0 (batch) is preserved
        # - Dimensions 1,2,3 (channels, height, width) are flattened
        # - Result: (N, 512*7*7) = (N, 25088)
        x = torch.flatten(x, 1)
        
        # Classification: Transform features to class predictions
        x = self.classifier(x)  # (N, 25088) -> (N, num_classes)
        return x
    

if __name__ == "__main__":
    """
    Example usage and testing of VGG11.
    
    This demonstrates:
    1. Model instantiation
    2. Input shape requirements
    3. Output shape verification
    """
    # Create model for ImageNet classification (1000 classes)
    model = VGG11(num_classes=1000)
    
    # Create a random input batch
    # Shape: (batch_size=1, channels=3, height=224, width=224)
    sample_input = torch.randn(1, 3, 224, 224)
    
    # Forward pass
    sample_output = model(sample_input)
    
    # Verify output shape
    print(f"Input shape:  {sample_input.shape}")   # torch.Size([1, 3, 224, 224])
    print(f"Output shape: {sample_output.shape}")  # torch.Size([1, 1000])
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # VGG11 has approximately 132 million parameters
    # Most parameters are in the fully connected layers!





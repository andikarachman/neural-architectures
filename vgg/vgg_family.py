
"""
VGG Family: Configurable Implementation of VGG Networks

This module provides a flexible implementation of the entire VGG family:
- VGG11: 8 conv layers + 3 FC layers = 11 weight layers
- VGG13: 10 conv layers + 3 FC layers = 13 weight layers
- VGG16: 13 conv layers + 3 FC layers = 16 weight layers
- VGG19: 16 conv layers + 3 FC layers = 19 weight layers

Key Features:
1. Configuration-driven architecture using cfg dictionaries
2. Optional Batch Normalization support (batch_norm parameter)
3. Customizable input channels and output classes
4. Unified implementation reducing code duplication

Architecture Encoding:
- Integers (64, 128, 256, 512): Number of output channels for Conv2d layer
- "M": MaxPooling operation (2x2, stride 2)
- Each conv layer uses 3x3 kernel with padding=1

Design Philosophy:
- Deeper networks (VGG16, VGG19) learn more complex representations
- All models share the same classifier head (3 FC layers)
- Simple, uniform structure makes it easy to understand and modify
"""

from typing import Union

import torch
import torch.nn as nn


# Configuration dictionaries for different VGG variants
# The deeper the network, the more convolutional layers
cfgs: dict[str, list[Union[str, int]]] = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def make_layers(cfg: list[Union[str, int]], in_channels: int = 3, batch_norm: bool = False) -> nn.Sequential:
    """
    Build the VGG feature extractor dynamically from configuration.
    
    This function constructs the convolutional backbone by parsing the configuration list.
    It supports optional Batch Normalization for improved training dynamics.
    
    Args:
        cfg: Configuration list defining the architecture
             - Integer values: Create Conv2d layer with that many output channels
             - "M": Insert MaxPool2d layer for spatial downsampling
        in_channels: Number of input channels (default: 3 for RGB images)
        batch_norm: If True, adds BatchNorm2d after each Conv2d layer (default: False)
    
    Returns:
        nn.Sequential: The constructed feature extraction layers
    
    Layer Order:
        Without BN: Conv2d -> ReLU -> [MaxPool2d if "M"]
        With BN:    Conv2d -> BatchNorm2d -> ReLU -> [MaxPool2d if "M"]
    
    Spatial Dimensions (with 224x224 input and 5 pools):
        224 -> 112 -> 56 -> 28 -> 14 -> 7
    
    Final output: (N, 512, 7, 7) for all VGG variants
    """
    layers = []
    c_in = in_channels
    
    for v in cfg:
        if v == "M":
            # MaxPool: Halves spatial dimensions (H, W) -> (H/2, W/2)
            # Provides translation invariance and reduces computation
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Convolutional layer: Extracts features using 3x3 filters
            c_out = int(v)
            conv2d = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
            
            if batch_norm:
                # Pattern: Conv -> BatchNorm -> ReLU
                # BatchNorm normalizes activations for faster, more stable training
                layers += [conv2d, nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            else:
                # Original VGG pattern: Conv -> ReLU
                layers += [conv2d, nn.ReLU(inplace=True)]
            
            c_in = c_out
    
    return nn.Sequential(*layers)

class VGG(nn.Module):
    """
    Unified VGG implementation supporting all variants (VGG11/13/16/19).
    
    This class provides a single implementation that can instantiate any VGG variant
    by simply changing the configuration name. Optional Batch Normalization support
    allows for faster training and better convergence.
    
    Architecture:
        Input (N, 3, 224, 224)
          ↓
        Feature Extractor (convolutional blocks)
          ↓
        Flatten (N, 512*7*7)
          ↓
        Classifier (3 fully connected layers)
          ↓
        Output (N, num_classes)
    """
    
    def __init__(self, cfg_name: str, num_classes: int = 1000, in_channels: int = 3, 
                 batch_norm: bool = False, dropout: float = 0.5) -> None:
        """
        Initialize VGG network.
        
        Args:
            cfg_name: Name of VGG variant ("VGG11", "VGG13", "VGG16", or "VGG19")
            num_classes: Number of output classes (default: 1000 for ImageNet)
            in_channels: Number of input channels (default: 3 for RGB)
            batch_norm: Whether to use Batch Normalization (default: False)
                       Setting to True creates VGG-BN variant
            dropout: Dropout probability in classifier (default: 0.5)
                    Can reduce to ~0.3 when using batch_norm=True
        
        Note:
            - batch_norm=False: Original VGG as described in the 2014 paper
            - batch_norm=True: VGG-BN variant with faster training and convergence
        """
        super().__init__()

        # Build feature extraction backbone from configuration
        self.features = make_layers(cfgs[cfg_name], in_channels=in_channels, batch_norm=batch_norm)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),               # (N, 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),                      # (N, 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),                # (N, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the VGG network.
        
        Data Flow:
        1. Feature extraction: Conv blocks progressively learn hierarchical features
        2. Flatten: Convert 3D feature maps to 1D feature vector
        3. Classification: Fully connected layers map features to class predictions
        
        Args:
            x: Input tensor of shape (N, C, H, W)
               Typically (N, 3, 224, 224) for ImageNet-style inputs
        
        Returns:
            Output tensor of shape (N, num_classes) with raw logits
        
        Note:
            If batch_norm=True, remember to call model.eval() before inference
            to use running statistics instead of batch statistics.
        """
        # Feature extraction: (N, 3, 224, 224) -> (N, 512, 7, 7)
        # Hierarchical learning: edges -> textures -> patterns -> parts -> objects
        x = self.features(x)
        
        # Flatten: (N, 512, 7, 7) -> (N, 25088)
        # Converts spatial features to a single feature vector per sample
        x = torch.flatten(x, 1)
        
        # Classification: (N, 25088) -> (N, num_classes)
        # Transforms features into class predictions
        x = self.classifier(x)
        return x

# Convenience functions for creating specific VGG variants
# These provide a cleaner API for instantiating models

def vgg11(num_classes: int = 1000, in_channels: int = 3, batch_norm: bool = False) -> VGG:
    """VGG11: 8 convolutional layers. Set batch_norm=True for VGG11-BN."""
    return VGG("VGG11", num_classes=num_classes, in_channels=in_channels, batch_norm=batch_norm)


def vgg13(num_classes: int = 1000, in_channels: int = 3, batch_norm: bool = False) -> VGG:
    """VGG13: 10 convolutional layers. Set batch_norm=True for VGG13-BN."""
    return VGG("VGG13", num_classes=num_classes, in_channels=in_channels, batch_norm=batch_norm)


def vgg16(num_classes: int = 1000, in_channels: int = 3, batch_norm: bool = False) -> VGG:
    """VGG16: 13 convolutional layers (most popular variant). Set batch_norm=True for VGG16-BN."""
    return VGG("VGG16", num_classes=num_classes, in_channels=in_channels, batch_norm=batch_norm)


def vgg19(num_classes: int = 1000, in_channels: int = 3, batch_norm: bool = False) -> VGG:
    """VGG19: 16 convolutional layers (deepest variant). Set batch_norm=True for VGG19-BN."""
    return VGG("VGG19", num_classes=num_classes, in_channels=in_channels, batch_norm=batch_norm)

if __name__ == "__main__":
    """
    Demonstration of VGG family with and without Batch Normalization.
    
    This example shows:
    1. How to create different VGG variants
    2. Comparison between original VGG and VGG-BN
    3. Parameter count differences
    4. Proper usage patterns
    """
    
    print("=" * 70)
    print("VGG Family - Architecture Comparison")
    print("=" * 70)
    
    # Create sample input
    x = torch.randn(1, 3, 224, 224)
    
    # Example 1: Original VGG16 (without Batch Normalization)
    print("\n[1] VGG16 (Original)")
    model_original = vgg16(num_classes=1000, batch_norm=False)
    model_original.eval()
    with torch.no_grad():
        y = model_original(x)
    params_original = sum(p.numel() for p in model_original.parameters())
    print(f"    Output shape: {y.shape}")
    print(f"    Parameters: {params_original:,}")
    
    # Example 2: VGG16 with Batch Normalization
    print("\n[2] VGG16-BN (With Batch Normalization)")
    model_bn = vgg16(num_classes=1000, batch_norm=True)
    model_bn.eval()
    with torch.no_grad():
        y_bn = model_bn(x)
    params_bn = sum(p.numel() for p in model_bn.parameters())
    bn_params = params_bn - params_original
    print(f"    Output shape: {y_bn.shape}")
    print(f"    Parameters: {params_bn:,}")
    print(f"    Additional BN params: {bn_params:,} (+{bn_params/params_original*100:.2f}%)")
    
    # Example 3: Compare all VGG variants
    print("\n[3] All VGG Variants - Parameter Count")
    variants = [
        ("VGG11", vgg11),
        ("VGG13", vgg13),
        ("VGG16", vgg16),
        ("VGG19", vgg19),
    ]
    
    for name, builder_fn in variants:
        model = builder_fn(num_classes=1000, batch_norm=False)
        params = sum(p.numel() for p in model.parameters())
        print(f"    {name}: {params:,} parameters")
    
    # Example 4: Custom configuration
    print("\n[4] Custom Configuration Example")
    print("    Creating VGG19-BN for CIFAR-10 (10 classes, 32x32 images):")
    # Note: For 32x32 images, you might want to reduce pooling layers or adjust architecture
    model_custom = vgg19(num_classes=10, in_channels=3, batch_norm=True)
    x_small = torch.randn(1, 3, 32, 32)
    model_custom.eval()
    with torch.no_grad():
        try:
            y_custom = model_custom(x_small)
            print(f"    Output shape: {y_custom.shape}")
        except RuntimeError as e:
            print(f"    Note: Standard VGG expects 224x224 input.")
            print(f"    For smaller images, architecture modifications needed.")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("  • Batch Normalization adds minimal parameters (~1-2% increase)")
    print("  • VGG19 has most parameters due to more convolutional layers")
    print("  • Most parameters are in the fully connected classifier layers")
    print("  • batch_norm=True recommended for faster training from scratch")
    print("  • Original VGG (batch_norm=False) when loading pretrained weights")
    print("=" * 70)
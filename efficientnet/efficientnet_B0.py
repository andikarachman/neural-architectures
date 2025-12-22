

import math
import torch
import torch.nn as nn   
import torch.nn.functional as F

class SiLU(nn.Module):
    """
    SiLU = Swish activation used in EfficientNet.
    PyTorch also provides nn.SiLU, but we keep this explicit.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    

def conv_bn_act(in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int) -> nn.Sequential:
    """
    Convolution + BatchNorm + Activation block.
    """
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        SiLU(),
    )


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation (SE) block as used in EfficientNet.
    
    A channel attention mechanism that adaptively recalibrates channel-wise 
    feature responses by explicitly modeling interdependencies between channels.
    
    Process:
    1. Squeeze: Global average pooling compresses spatial dimensions (H×W) 
       into a channel descriptor [N, C, 1, 1]
    2. Excitation: Two FC layers learn channel-wise attention weights:
       - FC1: C → C/reduce_ratio with ReLU (dimensionality reduction)
       - FC2: C/reduce_ratio → C with Sigmoid (produces weights [0,1])
    3. Scale: Multiply input features by learned attention weights
    
    Args:
        in_channels (int): Number of input channels
        reduce_ratio (int): Channel reduction ratio for bottleneck (default: 4)
    
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W) - same as input
    """
    def __init__(self, in_channels: int, reduce_ratio: int = 4) -> None:
        super().__init__()
        reduced_channels = in_channels // reduce_ratio
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Output size (N, C, 1, 1)
        
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        
        # Squeeze
        s = self.squeeze(x)     # (N, C, 1, 1)
        
        # Excitation
        s = self.excitation(s)  # (N, C, 1, 1) with values in [0, 1]
        
        # Scale
        return x * s            # Channel-wise scaling
    

class MBConvBlock(nn.Module):
    """
    MBConv: Mobile Inverted Bottleneck Convolution block.
    
    Core building block of EfficientNet that combines efficiency and expressiveness
    through depthwise separable convolutions and inverted bottleneck design.
    
    Architecture (3 stages + optional SE):
    1. Expansion: 1×1 conv increases channels by expansion_ratio (e.g., 6×)
       - Transforms narrow → wide (e.g., 24 → 144 channels)
    2. Depthwise: k×k depthwise conv captures spatial patterns efficiently
       - Each channel processed independently (groups = channels)
       - Applies stride for spatial downsampling if needed
    3. Projection: 1×1 conv reduces back to output channels (no activation)
       - Transforms wide → narrow (e.g., 144 → 24 channels)
       - Linear bottleneck preserves information
    4. SE Block (optional): Channel attention after depthwise conv
    5. Skip Connection: Residual added when in_channels == out_channels and stride == 1
    
    Why "Inverted" Bottleneck?
    - Traditional ResNet: Wide → Narrow → Wide (256 → 64 → 256)
    - MBConv (Inverted): Narrow → Wide → Narrow (24 → 144 → 24)
    - Keeps efficient narrow representations at input/output
    - Expands internally for richer transformations
    
    Efficiency:
    - Depthwise separable convs reduce params by ~8-9× vs standard convolutions
    - Narrow input/output reduces memory and computation
    - SE blocks add <2% parameters but improve accuracy
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size for depthwise conv (3 or 5)
        stride (int): Stride for depthwise conv (1 or 2)
        expansion_ratio (int): Channel expansion ratio (typically 1 or 6)
        se_ratio (float): SE reduction ratio (0.25 means reduce by 4×, 0 to disable)
    
    Shape:
        - Input: (N, in_channels, H, W)
        - Output: (N, out_channels, H/stride, W/stride)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, expansion_ratio: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * expansion_ratio

        # 1) Expansion phase (only if expansion_ratio != 1)
        if expansion_ratio != 1:
            self.expand_conv = conv_bn_act(in_channels, mid_channels,
                                           kernel_size=1, stride=1, groups=1)
        
        # 2) Depthwise convolution
        self.depthwise_conv = conv_bn_act(mid_channels, mid_channels,
                                          kernel_size=kernel_size, 
                                          stride=stride, groups=mid_channels)
        
        # 3) Squeeze-and-Excitation
        self.use_se = se_ratio > 0
        if self.use_se:
            self.se_block = SqueezeExcite(mid_channels, reduce_ratio=int(1/se_ratio))
        
        # 4) Projection phase
        self.project_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand_conv(x) if hasattr(self, 'expand_conv') else x    # (N, mid_channels, H, W)
        out = self.depthwise_conv(out)                                      # (N, mid_channels, H/stride, W/stride)
        if self.use_se:
            out = self.se_block(out)                                        # (N, mid_channels, H/stride, W/stride)
        out = self.project_conv(out)                                        # (N, out_channels, H/stride, W/stride)
        if self.use_residual:
            out = out + x                                                   # Residual connection
        return out
    


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0: Baseline model from the EfficientNet family.
    
    EfficientNet achieves state-of-the-art accuracy with fewer parameters by using
    compound scaling - uniformly scaling network depth, width, and resolution with
    a set of fixed scaling coefficients.
    
    Key Innovations:
    1. Compound Scaling: Balances depth/width/resolution instead of arbitrary scaling
       - Depth: Number of layers (controls capacity)
       - Width: Number of channels (controls feature richness)
       - Resolution: Input image size (controls fine-grained patterns)
    
    2. Neural Architecture Search (NAS): Base architecture (B0) found via NAS
       - Optimized for accuracy and FLOPS efficiency
       - Uses MBConv blocks with varying expansion ratios
    
    3. Core Components:
       - MBConv blocks: Mobile inverted bottleneck with depthwise separable convs
       - SE blocks: Channel attention (reduce ratio = 4)
       - SiLU/Swish activation: x * sigmoid(x) for smooth non-linearity
    
    EfficientNet-B0 Architecture:
    Stage | Block Type  | Channels | Layers | Stride | Kernel | Expansion | Resolution
    ------|-------------|----------|--------|--------|--------|-----------|------------
    1     | Conv        | 32       | 1      | 2      | 3×3    | -         | 112×112
    2     | MBConv1     | 16       | 1      | 1      | 3×3    | 1         | 112×112
    3     | MBConv6     | 24       | 2      | 2      | 3×3    | 6         | 56×56
    4     | MBConv6     | 40       | 2      | 2      | 5×5    | 6         | 28×28
    5     | MBConv6     | 80       | 3      | 2      | 3×3    | 6         | 14×14
    6     | MBConv6     | 112      | 3      | 1      | 5×5    | 6         | 14×14
    7     | MBConv6     | 192      | 4      | 2      | 5×5    | 6         | 7×7
    8     | MBConv6     | 320      | 1      | 1      | 3×3    | 6         | 7×7
    9     | Conv + Pool | 1280     | 1      | -      | 1×1    | -         | 7×7 → 1×1
    10    | FC          | num_classes | -   | -      | -      | -         | 1×1
    
    Model Stats (B0):
    - Parameters: ~5.3M
    - Input size: 224×224
    - Top-1 Accuracy (ImageNet): ~77.1%
    - FLOPS: ~0.39B
    
    Compared to ResNet-50:
    - 8.4× fewer parameters (5.3M vs 26M)
    - 6.1× fewer FLOPS
    - Similar or better accuracy
    
    Args:
        num_classes (int): Number of output classes (default: 1000 for ImageNet)
        dropout_rate (float): Dropout rate before classifier (default: 0.2)
    
    Shape:
        - Input: (N, 3, 224, 224)
        - Output: (N, num_classes)
    
    Reference:
        "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
        Mingxing Tan, Quoc V. Le (ICML 2019)
        https://arxiv.org/abs/1905.11946
    """
    def __init__(self, num_classes: int = 1000, dropout_rate: float = 0.2) -> None:
        super().__init__()

        # Stem: Initial Conv Layer
        self.stem = conv_bn_act(3, 32, kernel_size=3, stride=2, groups=1)  # (N, 32, 112, 112)

        # EfficientNet-B0 MBConv Blocks Configuration
        # Format: (in_channels, out_channels, kernel_size, stride, expansion_ratio, num_layers)
        blocks_cfg = [
            (32,   16, 3, 1, 1, 1),  # Stage 2
            (16,   24, 3, 2, 6, 2),  # Stage 3
            (24,   40, 5, 2, 6, 2),  # Stage 4
            (40,   80, 3, 2, 6, 3),  # Stage 5
            (80,   112,5, 1, 6, 3),  # Stage 6
            (112, 192,5, 2, 6, 4),  # Stage 7
            (192, 320,3, 1, 6, 1),  # Stage 8
        ]

        layers = []
        for in_c, out_c, k_size, stride, exp_ratio, n_layers in blocks_cfg:
            for i in range(n_layers):
                s = stride if i == 0 else 1  # Only first block in stage uses stride
                layers.append(MBConvBlock(in_c if i == 0 else out_c,
                                          out_c,
                                          kernel_size=k_size,
                                          stride=s,
                                          expansion_ratio=exp_ratio,
                                          se_ratio=0.25))
        
        self.blocks = nn.Sequential(*layers)

        # Head: Final Conv Layer before Pooling
        self.head_conv = conv_bn_act(320, 1280, kernel_size=1, stride=1, groups=1)  # (N, 1280, 7, 7)

        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1280, num_classes)

        # Initialize

    
    
"""
EfficientNet-B0: Compound Scaling for Efficient Neural Networks

EfficientNet was introduced in "EfficientNet: Rethinking Model Scaling for 
Convolutional Neural Networks" by Mingxing Tan and Quoc V. Le (ICML 2019).

THE CORE PROBLEM: INEFFICIENT SCALING
======================================
Traditional Approach to Improving Accuracy:
- Scale up depth (more layers) → ResNet-18 to ResNet-152
- Scale up width (more channels) → ResNet-50 to WideResNet
- Scale up resolution (larger images) → 224×224 to 299×299

Problem: Scaling these dimensions arbitrarily is inefficient and suboptimal
- Only scaling depth: Diminishing returns, harder to train very deep networks
- Only scaling width: Wide but shallow networks can't capture complex patterns
- Only scaling resolution: Larger images without more capacity waste computation

THE SOLUTION: COMPOUND SCALING
===============================
EfficientNet scales all three dimensions (depth, width, resolution) together
using a simple compound coefficient φ (phi):

    depth:      d = α^φ
    width:      w = β^φ  
    resolution: r = γ^φ
    
    Constraint: α · β² · γ² ≈ 2  (FLOPS roughly double for each φ increment)
    where α ≥ 1, β ≥ 1, γ ≥ 1

For EfficientNet-B0 to B7:
- B0: φ = 0 (baseline, found via Neural Architecture Search)
- B1: φ = 1, scales all dimensions proportionally
- B2: φ = 2, continues scaling
- ... up to B7

Key Insight: Balanced scaling is more efficient than arbitrary scaling

EFFICIENTNET INNOVATIONS:
=========================
1. COMPOUND SCALING METHOD
   - Principled way to scale networks for any resource constraint
   - Achieves better accuracy-efficiency trade-off than ad-hoc scaling
   - Grid search found optimal: α=1.2, β=1.1, γ=1.15 for EfficientNet

2. NEURAL ARCHITECTURE SEARCH (NAS) FOR BASELINE
   - B0 architecture discovered via NAS optimizing accuracy and FLOPS
   - Objective: maximize accuracy while keeping FLOPS ≤ target
   - Search space: MBConv blocks with varying kernel sizes and expansion ratios

3. MOBILE INVERTED BOTTLENECK CONVOLUTION (MBConv)
   - Inherits from MobileNetV2 but with enhancements
   - Inverted residual structure: Narrow → Wide → Narrow
   - Depthwise separable convolutions for efficiency
   - Squeeze-and-Excitation (SE) blocks for channel attention

4. SWISH/SiLU ACTIVATION
   - Smooth activation: f(x) = x · sigmoid(x)
   - Better than ReLU for deeper networks
   - Allows gradients to flow more smoothly

ARCHITECTURE OVERVIEW (B0):
===========================
Stage | Block    | Channels | #Layers | Stride | Kernel | Expansion | Resolution
------|----------|----------|---------|--------|--------|-----------|------------
1     | Conv3×3  | 32       | 1       | 2      | 3×3    | -         | 112×112
2     | MBConv1  | 16       | 1       | 1      | 3×3    | 1         | 112×112
3     | MBConv6  | 24       | 2       | 2      | 3×3    | 6         | 56×56
4     | MBConv6  | 40       | 2       | 2      | 5×5    | 6         | 28×28
5     | MBConv6  | 80       | 3       | 2      | 3×3    | 6         | 14×14
6     | MBConv6  | 112      | 3       | 1      | 5×5    | 6         | 14×14
7     | MBConv6  | 192      | 4       | 2      | 5×5    | 6         | 7×7
8     | MBConv6  | 320      | 1       | 1      | 3×3    | 6         | 7×7
9     | Conv1×1  | 1280     | 1       | -      | 1×1    | -         | 7×7
10    | Pool+FC  | classes  | 1       | -      | -      | -         | 1×1

Total Layers: 1 stem + 16 MBConv blocks + 1 head conv = 18 convolutional layers
Total Parameters: ~5.3M (much smaller than ResNet-50's 26M)

KEY ARCHITECTURAL COMPONENTS:
=============================

1. MBConv Block (Mobile Inverted Bottleneck Conv):
   Input (narrow) 
     ↓
   1×1 Conv - Expansion (narrow → wide, 6× channels)
     ↓
   Depthwise Conv - Spatial filtering (k×k, each channel separately)
     ↓
   SE Block - Channel attention (optional but default)
     ↓
   1×1 Conv - Projection (wide → narrow, no activation)
     ↓
   Skip Connection (if input/output dims match)
     ↓
   Output (narrow)

2. Squeeze-and-Excitation (SE) Block:
   - Global average pooling: Squeeze spatial info (H×W → 1×1)
   - FC layers: Learn channel importance (C → C/4 → C)
   - Sigmoid: Output attention weights [0,1]
   - Scale: Multiply features by attention weights
   - Adds <2% params but improves accuracy significantly

3. Depthwise Separable Convolution:
   - Standard conv: O(k² · C_in · C_out · H · W)
   - Depthwise + Pointwise: O(k² · C_in · H · W + C_in · C_out · H · W)
   - Reduction: ~8-9× fewer parameters and computations
   - Trades off very slight accuracy for massive efficiency

WHY EFFICIENTNET WORKS:
========================
1. BALANCED SCALING: All dimensions scale together harmoniously
2. OPTIMAL BASELINE: NAS finds efficient base architecture (B0)
3. PARAMETER EFFICIENCY: Depthwise separable convs reduce parameters
4. CHANNEL ATTENTION: SE blocks focus on important features
5. SMOOTH ACTIVATIONS: Swish/SiLU enables better gradient flow

EFFICIENTNET FAMILY COMPARISON:
================================
Model  | Params  | FLOPS  | Input Size | ImageNet Top-1 | vs ResNet-50
-------|---------|--------|------------|----------------|----------------
B0     | 5.3M    | 0.39B  | 224×224   | 77.1%          | 5× smaller
B1     | 7.8M    | 0.70B  | 240×240   | 79.1%          | Better, smaller
B2     | 9.2M    | 1.0B   | 260×260   | 80.1%          | Better, smaller
B3     | 12M     | 1.8B   | 300×300   | 81.6%          | Better, smaller
B4     | 19M     | 4.2B   | 380×380   | 82.9%          | Similar size
B5     | 30M     | 9.9B   | 456×456   | 83.6%          | Larger, better
B6     | 43M     | 19B    | 528×528   | 84.0%          | Much larger
B7     | 66M     | 37B    | 600×600   | 84.3%          | Much larger

PRACTICAL IMPACT:
=================
- Mobile/Edge Deployment: B0-B2 run efficiently on mobile devices
- Cloud Deployment: B3-B5 balance accuracy and cost
- Research: B6-B7 push state-of-the-art boundaries
- Transfer Learning: Excellent features for downstream tasks

DESIGN PRINCIPLES:
==================
1. Efficiency First: Optimize for accuracy per FLOP, not just accuracy
2. Systematic Scaling: Use compound scaling instead of ad-hoc tuning
3. Search + Scale: Find good baseline (B0), then scale systematically
4. Mobile-Friendly: Designed to work on resource-constrained devices
5. Universal: Same architecture family works across resource budgets

Reference:
    Mingxing Tan and Quoc V. Le. "EfficientNet: Rethinking Model Scaling 
    for Convolutional Neural Networks." ICML 2019.
    https://arxiv.org/abs/1905.11946
"""

import math
import torch
import torch.nn as nn   
import torch.nn.functional as F


class SiLU(nn.Module):
    """
    SiLU (Sigmoid Linear Unit) / Swish Activation Function.
    
    Formula: f(x) = x · sigmoid(x) = x / (1 + e^(-x))
    
    Properties:
    - Smooth, non-monotonic activation function
    - Self-gating: output depends on both input value and its magnitude
    - Unbounded above, bounded below (approaches 0 as x → -∞)
    - Better than ReLU for deeper networks in many tasks
    - Computational cost: slightly higher than ReLU (requires sigmoid)
    
    Why it works:
    - Smooth gradients enable better optimization
    - Non-monotonic property allows some negative values through
    - Self-gating mechanism provides implicit attention
    
    Note: PyTorch provides nn.SiLU built-in, but we implement explicitly
          for educational purposes and clarity.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

def conv_bn_act(in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int) -> nn.Sequential:
    """
    Standard building block: Convolution → Batch Normalization → Activation.
    
    This is a fundamental pattern used throughout EfficientNet for consistent
    feature transformation with normalization and non-linearity.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolving kernel (typically 1, 3, or 5)
        stride: Stride of convolution (1 for same size, 2 for downsampling)
        groups: Number of groups for grouped convolution
                - groups=1: Standard convolution (all-to-all connectivity)
                - groups=in_channels: Depthwise convolution (channel-wise)
    
    Returns:
        nn.Sequential: Conv2d → BatchNorm2d → SiLU activation
    
    Design Choices:
        - bias=False: BatchNorm has its own bias term, conv bias is redundant
        - padding: Automatically calculated to maintain spatial dimensions when stride=1
        - SiLU activation: Smooth non-linearity preferred in EfficientNet
    """
    padding = (kernel_size - 1) // 2  # Maintains spatial size when stride=1
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
            (32,  16,  3, 1, 1, 1),  # Stage 2
            (16,  24,  3, 2, 6, 2),  # Stage 3
            (24,  40,  5, 2, 6, 2),  # Stage 4
            (40,  80,  3, 2, 6, 3),  # Stage 5
            (80,  112, 5, 1, 6, 3),  # Stage 6
            (112, 192, 5, 2, 6, 4),  # Stage 7
            (192, 320, 3, 1, 6, 1),  # Stage 8
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                # (N, 32, 112, 112)
        x = self.blocks(x)              # (N, 320, 7, 7)
        x = self.head_conv(x)           # (N, 1280, 7, 7)
        x = F.adaptive_avg_pool2d(x, 1) # (N, 1280, 1, 1)
        x = torch.flatten(x, 1)         # Flatten to (N, 1280)
        x = self.dropout(x)             # Apply dropout
        x = self.classifier(x)          # (N, num_classes)
        return x

if __name__ == "__main__":
    print("=" * 70)
    print("EfficientNet-B0 (ImageNet-style)")
    print("=" * 70)

    model = EfficientNetB0(num_classes=1000)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    print("\n[1] Input/Output")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    print("\n[2] Model Summary")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


    
    
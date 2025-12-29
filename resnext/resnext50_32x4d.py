"""
ResNeXt: Aggregated Residual Transformations for Deep Neural Networks

ResNeXt was introduced in "Aggregated Residual Transformations for Deep Neural 
Networks" by Xie et al. (UC San Diego, Facebook AI Research, 2017).

Winner of ILSVRC 2016 classification task.

THE INNOVATION: CARDINALITY AS A NEW DIMENSION
==============================================
Deep learning has three fundamental dimensions:
1. Depth (number of layers)
2. Width (number of channels)
3. ??? (the missing dimension)

ResNeXt introduces CARDINALITY (C): the size of the set of transformations
"We use the term 'cardinality' to refer to the size of the set of 
transformations. We show that cardinality is an essential dimension and can be 
more effective than the dimensions of width and depth."

Key Insight: Instead of making networks deeper or wider, we can make them 
            "wider in a new dimension" by using multiple parallel paths.

HISTORICAL CONTEXT
==================
Evolution of network design:
1. AlexNet (2012): Stack convolutions deeper
2. VGG (2014): Very deep, homogeneous architecture
3. Inception (2014): "Network in Network", heterogeneous multi-branch
4. ResNet (2015): Very deep with residual connections
5. ResNeXt (2016): Homogeneous multi-branch (best of both worlds!)

The Dilemma:
- Inception: High accuracy but complex, hard to adapt
- ResNet: Simple, modular, but limited by width/depth tradeoffs
- Solution: ResNeXt combines Inception's multi-path with ResNet's simplicity

ARCHITECTURE: SPLIT-TRANSFORM-MERGE
====================================

Traditional ResNet Bottleneck:
    Input (256)
       ↓
    1×1 Conv (256 → 64)     [Compress]
       ↓
    3×3 Conv (64 → 64)      [Transform]
       ↓
    1×1 Conv (64 → 256)     [Expand]
       ↓
    + [Residual]
       ↓
    Output (256)

ResNeXt Bottleneck (32 groups, 4d per group):
    Input (256)
       ↓
    1×1 Conv (256 → 128)    [Compress]
       ↓
    ╔════════════════════════════════════╗
    ║  3×3 Grouped Conv (32 groups)     ║
    ║  Group 1: 4 channels → 4 channels ║
    ║  Group 2: 4 channels → 4 channels ║
    ║  ...                               ║
    ║  Group 32: 4 channels → 4 channels║
    ║                                    ║
    ║  Total: 128 → 128                 ║
    ║  (32 groups × 4 channels)         ║
    ╚════════════════════════════════════╝
       ↓
    1×1 Conv (128 → 256)    [Expand]
       ↓
    + [Residual]
       ↓
    Output (256)

Key Parameters:
- Cardinality C = 32 (number of groups)
- Width d = 4 (channels per group)
- Total width = C × d = 128

EQUIVALENT FORMULATIONS
========================
The ResNeXt block can be understood in three equivalent ways:

1. Grouped Convolutions (Implemented):
   y = x + Σᵢ₌₁³² Tᵢ(x)
   Where each Tᵢ is a narrow (4-channel) transformation
   
   Implementation: Use groups=32 in nn.Conv2d

2. Split-Concat (Conceptual):
   - Split input into 32 groups (4 channels each)
   - Apply 3×3 conv to each group independently
   - Concatenate results
   
   Like Inception, but all paths are identical!

3. Multi-Path (Mathematical):
   - 32 parallel paths with shared structure
   - Each path processes a subset of channels
   - Aggregate via element-wise addition
   
   Similar to Inception but homogeneous topology

All three are mathematically equivalent but offer different perspectives.

DESIGN PRINCIPLES
=================

1. HOMOGENEOUS TOPOLOGY:
   All paths have the same structure (same hyperparameters)
   - Easy to design: only one template
   - Easy to scale: change cardinality uniformly
   - Easy to analyze: no path-specific tuning
   
   vs. Inception: Heterogeneous (different filter sizes per path)

2. SPLIT-TRANSFORM-MERGE STRATEGY:
   - Split: 1×1 conv divides into C groups
   - Transform: Each group processed independently
   - Merge: 1×1 conv aggregates all groups
   
   Similar to: Inception (multi-branch), ResNet (bottleneck)

3. COMPLEXITY-PRESERVING:
   ResNeXt is designed to have similar complexity to ResNet
   
   ResNet-50:
   - 1×1: 256 → 64
   - 3×3: 64 → 64 (standard)
   - 1×1: 64 → 256
   - FLOPs: 256×64 + 64×64×9 + 64×256
   
   ResNeXt-50 32×4d:
   - 1×1: 256 → 128
   - 3×3: 128 → 128 (grouped, 32 groups)
   - 1×1: 128 → 256
   - FLOPs: 256×128 + 128×128×9/32 + 128×256
   
   Similar FLOPs but better accuracy!

4. WIDTH vs CARDINALITY:
   Increasing cardinality is more effective than increasing width
   
   Experiment (same FLOPs):
   - Width 64, C=1:  22.2% error
   - Width 40, C=2:  21.7% error
   - Width 28, C=4:  21.4% error
   - Width 14, C=8:  21.3% error
   - Width 4, C=32:  21.2% error ← Best!
   
   Lesson: "Cardinality is a more effective dimension than width"

GROUPED CONVOLUTIONS
=====================
Standard convolution:
- Input: (B, C_in, H, W)
- Output: (B, C_out, H, W)
- Each output channel sees ALL input channels
- Parameters: K × K × C_in × C_out

Grouped convolution (groups=G):
- Split input into G groups: C_in / G channels each
- Split output into G groups: C_out / G channels each
- Each output group only sees its corresponding input group
- Parameters: K × K × (C_in/G) × C_out
  = (K × K × C_in × C_out) / G
- FLOPs reduced by factor of G!

Example: 128 channels, 32 groups
- Standard: Each of 128 outputs sees all 128 inputs
- Grouped: Each of 4 outputs sees only 4 inputs (within its group)
- 32 groups operate in parallel
- Much more efficient!

Benefits:
✓ Reduces computation (fewer connections)
✓ Reduces parameters (less overfitting)
✓ Increases paths (more diverse features)
✓ Maintains capacity (parallel processing)

RESNEXT-50 32x4d ARCHITECTURE
==============================
"32x4d" means: 32 groups (cardinality) × 4 channels per group (width)

Overall Structure (same as ResNet-50):
1. Stem: 7×7 conv, stride 2 → max pool
2. Stage 1: 3 blocks  (output: 56×56, 256 channels)
3. Stage 2: 4 blocks  (output: 28×28, 512 channels)
4. Stage 3: 6 blocks  (output: 14×14, 1024 channels)
5. Stage 4: 3 blocks  (output: 7×7, 2048 channels)
6. Head: Global avg pool → FC 1000

Block Structure (ResNeXt Bottleneck):
- 1×1 conv: C_in → 128 (compression)
- 3×3 grouped conv: 128 → 128 (groups=32, 4 channels per group)
- 1×1 conv: 128 → C_out (expansion)
- Batch norm + ReLU after each conv
- Residual connection

Width Calculation:
For cardinality C and width d:
- Middle channels = C × d = 32 × 4 = 128
- Stage 1: 32 × 4 = 128
- Stage 2: 32 × 8 = 256
- Stage 3: 32 × 16 = 512
- Stage 4: 32 × 32 = 1024

CARDINALITY CONFIGURATIONS
===========================
Different ResNeXt variants:
- ResNeXt-50 32×4d: 32 groups, 4 channels/group, 25M params
- ResNeXt-101 32×8d: 32 groups, 8 channels/group, 89M params
- ResNeXt-101 64×4d: 64 groups, 4 channels/group, 83M params

General pattern:
- Increasing C (cardinality): Better but diminishing returns
- Increasing d (width): Less effective than increasing C
- Sweet spot: C=32 for most applications

PERFORMANCE HIGHLIGHTS
=======================
ImageNet-1K (224×224):
- ResNeXt-50 32×4d: 77.6% → 81.2% (with improved recipe)
- ResNeXt-101 32×8d: 79.3% → 82.8% (with improved recipe)
- ResNeXt-101 64×4d: 83.2% (with improved recipe)

Key Observations:
- Matches ResNet-101 with only 50 layers
- 2× better than ResNet-50 with similar complexity
- Scales well: deeper models improve more
- Strong transfer to detection/segmentation

COCO Object Detection:
- Faster R-CNN + ResNeXt-101: 41.2 mAP
- Better than ResNet-101: 38.1 mAP

ADVANTAGES
==========
✓ Simple, modular design (like ResNet)
✓ Easy to adapt (change cardinality uniformly)
✓ Better accuracy than ResNet (same complexity)
✓ Strong scaling properties (deeper is better)
✓ Excellent transfer learning
✓ Efficient grouped convolutions
✓ Homogeneous topology (no manual tuning)

LIMITATIONS
===========
✗ Grouped convolutions not optimized on all hardware
✗ Cardinality must divide channel count evenly
✗ Memory bandwidth can be bottleneck
✗ Benefits saturate at very high cardinality (C > 64)

IMPLEMENTATION NOTES
====================
1. Grouped convolutions use groups parameter in nn.Conv2d
2. Width calculation: width = int(planes * (base_width / 64)) * groups
3. Base width is the width per group (e.g., 4 for 32×4d)
4. Groups apply only to 3×3 convolution (not 1×1)

KEY TAKEAWAYS
=============
1. Cardinality is a new, effective dimension for scaling
2. Increasing cardinality > increasing width (same FLOPs)
3. Homogeneous multi-path = simplicity + accuracy
4. Grouped convolutions = efficient implementation
5. ResNeXt scales better than ResNet

Historical Impact:
- Influenced ShuffleNet, MobileNet (group convolutions)
- Inspired Neural Architecture Search (new dimensions)
- Showed importance of parallel paths
- Demonstrated: Simple rules > complex engineering

Reference:
    Saining Xie et al. "Aggregated Residual Transformations for Deep 
    Neural Networks." CVPR 2017.
    https://arxiv.org/abs/1611.05431
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable, Type


def conv3x3(
    in_planes: int, 
    out_planes: int, 
    stride: int = 1, 
    groups: int = 1, 
    dilation: int = 1
) -> nn.Conv2d:
    """
    3×3 convolution with padding.
    
    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        stride: Stride for convolution (1 or 2)
        groups: Number of groups for grouped convolution
        dilation: Dilation rate for dilated convolution
    
    Returns:
        3×3 Conv2d layer
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    1×1 convolution (pointwise).
    
    Used for:
    - Changing channel dimensions
    - Downsampling (when stride=2)
    - No spatial mixing (only channel mixing)
    
    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        stride: Stride for downsampling (1 or 2)
    
    Returns:
        1×1 Conv2d layer
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt Bottleneck Block with grouped convolutions.
    
    Structure:
    1. 1×1 conv: Compress channels
    2. 3×3 grouped conv: Transform with cardinality
    3. 1×1 conv: Expand channels
    4. Residual connection
    
    The grouped convolution is the key innovation:
    - Splits channels into 'groups' independent groups
    - Each group has 'base_width' channels
    - Total middle channels = groups × base_width
    
    Example: ResNeXt-50 32×4d Layer 1 Block
    - Input: inplanes=256, planes=64, groups=32, base_width=4, expansion=4
    - Width calculation: width = int(64 × (4/64)) × 32 = 4 × 32 = 128
    
    Dimensionality Flow:
    1. Input:        (B, 256, H, W)      [inplanes=256]
    2. 1×1 conv:     (B, 128, H, W)      [compress to width=128]
    3. 3×3 grouped:  (B, 128, H', W')    [32 groups × 4 channels, stride affects H,W]
    4. 1×1 conv:     (B, 256, H', W')    [expand to planes×expansion=64×4=256]
    5. + Residual:   (B, 256, H', W')    [add input, output=inplanes]
    
    Width scaling across stages (groups=32, base_width=4):
    - Stage 1 (planes=64):  width = int(64×(4/64))×32  = 4×32  = 128  → output: 256
    - Stage 2 (planes=128): width = int(128×(4/64))×32 = 8×32  = 256  → output: 512
    - Stage 3 (planes=256): width = int(256×(4/64))×32 = 16×32 = 512  → output: 1024
    - Stage 4 (planes=512): width = int(512×(4/64))×32 = 32×32 = 1024 → output: 2048
    
    Args:
        inplanes: Number of input channels
        planes: Base number of output channels (will be multiplied by expansion)
        stride: Stride for downsampling (1 or 2)
        downsample: Downsample module for residual path (if dimensions change)
        groups: Number of groups (cardinality C)
        base_width: Number of channels per group (width d)
        dilation: Dilation rate for atrous convolution
        norm_layer: Normalization layer (default: BatchNorm2d)
    """
    
    # Expansion factor: final output has 4× more channels than planes
    # Example: planes=64 → output=256
    expansion: int = 4
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Calculate width of middle layer
        # For ResNeXt-50 32×4d: planes=64, base_width=4, groups=32
        # width = int(64 * (4/64)) * 32 = int(64 * 0.0625) * 32 = 4 * 32 = 128
        width = int(planes * (base_width / 64.0)) * groups
        
        # ========== BOTTLENECK LAYERS ==========
        
        # 1. First 1×1 convolution: Compress to 'width' channels
        # This reduces dimensionality before expensive 3×3 conv
        # Example: 256 → 128
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        
        # 2. 3×3 grouped convolution: Core transformation
        # This is where cardinality comes in!
        # groups=32 means 32 parallel paths of 4 channels each
        # Example: 128 → 128 (split into 32 groups of 4)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        
        # 3. Second 1×1 convolution: Expand to output channels
        # Restores full dimensionality
        # Example: 128 → 256 (planes * expansion = 64 * 4)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        
        # Activation function (applied after BN)
        self.relu = nn.ReLU(inplace=True)
        
        # Downsample module for residual path (when dimensions change)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNeXt bottleneck.
        
        Args:
            x: (B, inplanes, H, W) input tensor
        
        Returns:
            (B, planes * expansion, H', W') output tensor
            where H' = H / stride, W' = W / stride
        """
        # Save input for residual connection
        identity = x
        
        # ========== MAIN PATH ==========
        
        # 1. Compress: 1×1 conv + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 2. Transform: 3×3 grouped conv + BN + ReLU
        # This is where split-transform happens
        # Each of 32 groups processes its 4 channels independently
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # 3. Expand: 1×1 conv + BN (no ReLU yet)
        # Merge happens implicitly (all groups concatenated)
        out = self.conv3(out)
        out = self.bn3(out)
        
        # ========== RESIDUAL CONNECTION ==========
        
        # Adjust identity if dimensions changed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual (element-wise addition)
        out += identity
        
        # Final ReLU activation
        out = self.relu(out)
        
        return out


class ResNeXt(nn.Module):
    """
    ResNeXt: Aggregated Residual Transformations.
    
    Architecture follows ResNet structure but uses ResNeXt bottleneck blocks
    with grouped convolutions for increased cardinality.
    
    Structure:
    1. Stem: 7×7 conv (stride 2) + max pool → 64 channels, 56×56
    2. Layer1: 3 blocks → 256 channels, 56×56
    3. Layer2: 4 blocks → 512 channels, 28×28 (stride 2)
    4. Layer3: 6 blocks → 1024 channels, 14×14 (stride 2)
    5. Layer4: 3 blocks → 2048 channels, 7×7 (stride 2)
    6. Head: Global avg pool + FC → num_classes
    
    Args:
        block: Building block class (ResNeXtBottleneck)
        layers: Number of blocks per stage [3, 4, 6, 3] for ResNeXt-50
        num_classes: Number of output classes
        zero_init_residual: Initialize final BN in each block to 0
        groups: Number of groups (cardinality, e.g., 32)
        width_per_group: Channels per group (e.g., 4 for 32×4d)
        replace_stride_with_dilation: Use dilated convolutions instead of stride
        norm_layer: Normalization layer (default: BatchNorm2d)
    """
    
    def __init__(
        self,
        block: Type[ResNeXtBottleneck],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        # Initial number of channels (after stem)
        self.inplanes = 64
        
        # Dilation rate for atrous convolution
        self.dilation = 1
        
        # Replace stride with dilation in stages 2-4 if specified
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        
        # Store groups and width for building blocks
        self.groups = groups
        self.base_width = width_per_group
        
        # ========== STEM ==========
        # Initial layers to process RGB image
        # 7×7 conv with stride 2: 224×224 → 112×112
        self.conv1 = nn.Conv2d(
            3, self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # Max pooling with stride 2: 112×112 → 56×56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ========== RESIDUAL STAGES ==========
        # Four stages with increasing channels and decreasing resolution
        
        # Stage 1: 56×56, 256 channels (64 * 4)
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        # Stage 2: 28×28, 512 channels (128 * 4)
        self.layer2 = self._make_layer(
            block, 128, layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        
        # Stage 3: 14×14, 1024 channels (256 * 4)
        self.layer3 = self._make_layer(
            block, 256, layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1]
        )
        
        # Stage 4: 7×7, 2048 channels (512 * 4)
        self.layer4 = self._make_layer(
            block, 512, layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2]
        )
        
        # ========== CLASSIFICATION HEAD ==========
        # Global average pooling: 7×7 → 1×1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer for classification
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # ========== WEIGHT INITIALIZATION ==========
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for conv layers
                # mode='fan_out': preserve variance in forward pass
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # Initialize BN with weight=1, bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        # Makes residual branch start as identity
        # Improves training by 0.2-0.3% (from ResNet paper)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNeXtBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
    
    def _make_layer(
        self,
        block: Type[ResNeXtBottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False
    ) -> nn.Sequential:
        """
        Build a residual stage with multiple blocks.
        
        Args:
            block: Block class (ResNeXtBottleneck)
            planes: Base number of channels for this stage
            blocks: Number of blocks in this stage
            stride: Stride for first block (for downsampling)
            dilate: Use dilation instead of stride
        
        Returns:
            Sequential module containing all blocks
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        # Update dilation if requested
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # Create downsample module if dimensions change
        # This happens at the start of each stage (except first)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )
        
        layers = []
        
        # First block (may downsample)
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer
            )
        )
        
        # Update inplanes for subsequent blocks
        self.inplanes = planes * block.expansion
        
        # Remaining blocks (no downsampling)
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )
        
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Internal forward pass implementation.
        
        Args:
            x: (B, 3, H, W) input images (typically 224×224)
        
        Returns:
            (B, num_classes) logits
        """
        # Stem: 224×224 → 56×56
        x = self.conv1(x)       # 224×224 → 112×112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 112×112 → 56×56
        
        # Residual stages
        x = self.layer1(x)      # 56×56, 256 channels
        x = self.layer2(x)      # 28×28, 512 channels
        x = self.layer3(x)      # 14×14, 1024 channels
        x = self.layer4(x)      # 7×7, 2048 channels
        
        # Classification head
        x = self.avgpool(x)     # 7×7 → 1×1
        x = torch.flatten(x, 1) # (B, 2048)
        x = self.fc(x)          # (B, num_classes)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, 3, 224, 224) input images
        
        Returns:
            (B, num_classes) logits
        """
        return self._forward_impl(x)


# ========== MODEL CONSTRUCTORS ==========

def resnext50_32x4d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    """
    ResNeXt-50 32×4d model.
    
    Configuration:
    - 50 layers deep (3 + 4 + 6 + 3 = 16 blocks × 3 layers + stem)
    - Cardinality C = 32 (32 groups)
    - Width d = 4 (4 channels per group)
    - Middle channels = C × d = 128
    
    Architecture:
    - Stem: 7×7 conv + max pool
    - Stage 1: 3 blocks, 256 channels
    - Stage 2: 4 blocks, 512 channels
    - Stage 3: 6 blocks, 1024 channels
    - Stage 4: 3 blocks, 2048 channels
    - Head: Global avg pool + FC
    
    Parameters: ~25M
    FLOPs: ~4.2G
    
    Performance:
    - ImageNet-1K: 77.6% → 81.2% (with improved training)
    - Better than ResNet-50 (76.1%) with similar complexity
    
    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments for ResNeXt
    
    Returns:
        ResNeXt-50 32×4d model
    """
    return ResNeXt(
        ResNeXtBottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        groups=32,
        width_per_group=4,
        **kwargs
    )


def resnext101_32x8d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    """
    ResNeXt-101 32×8d model.
    
    Configuration:
    - 101 layers deep (3 + 4 + 23 + 3 = 33 blocks × 3 layers + stem)
    - Cardinality C = 32 (32 groups)
    - Width d = 8 (8 channels per group)
    - Middle channels = C × d = 256
    
    Parameters: ~89M
    FLOPs: ~16.4G
    
    Performance:
    - ImageNet-1K: 79.3% → 82.8% (with improved training)
    - Competitive with EfficientNet-B3
    
    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments for ResNeXt
    
    Returns:
        ResNeXt-101 32×8d model
    """
    return ResNeXt(
        ResNeXtBottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
        groups=32,
        width_per_group=8,
        **kwargs
    )


def resnext101_64x4d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    """
    ResNeXt-101 64×4d model.
    
    Configuration:
    - 101 layers deep (3 + 4 + 23 + 3 = 33 blocks × 3 layers + stem)
    - Cardinality C = 64 (64 groups)
    - Width d = 4 (4 channels per group)
    - Middle channels = C × d = 256
    
    Parameters: ~83M
    FLOPs: ~15.5G
    
    Performance:
    - ImageNet-1K: 83.2% (with improved training)
    - Higher cardinality shows benefits
    
    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments for ResNeXt
    
    Returns:
        ResNeXt-101 64×4d model
    """
    return ResNeXt(
        ResNeXtBottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
        groups=64,
        width_per_group=4,
        **kwargs
    )


if __name__ == "__main__":
    # ========== DEMONSTRATION ==========
    print("=" * 70)
    print("ResNeXt: Aggregated Residual Transformations")
    print("=" * 70)
    
    # Create ResNeXt-50 32×4d model
    model = resnext50_32x4d(num_classes=1000)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nResNeXt-50 32×4d Architecture:")
    print(f"  Configuration: 32 groups × 4 channels = 128")
    print(f"  Layers: [3, 4, 6, 3] = 16 blocks")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Example input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output logits sample: {output[0, :5]}")
    
    # Show block structure
    print("\n" + "=" * 70)
    print("ResNeXt Bottleneck Block")
    print("=" * 70)
    
    block = ResNeXtBottleneck(
        inplanes=256,
        planes=64,
        groups=32,
        base_width=4
    )
    
    x_block = torch.randn(1, 256, 56, 56)
    print(f"Input:  {x_block.shape}")
    
    with torch.no_grad():
        y_block = block(x_block)
    
    print(f"Output: {y_block.shape}")
    
    # Calculate middle channels
    width = int(64 * (4 / 64.0)) * 32
    print(f"\nMiddle channels calculation:")
    print(f"  planes × (base_width / 64) × groups")
    print(f"  = 64 × (4 / 64) × 32")
    print(f"  = 64 × 0.0625 × 32")
    print(f"  = {width}")
    
    print(f"\nBlock structure:")
    print(f"  1. 1×1 conv: 256 → {width} (compress)")
    print(f"  2. 3×3 grouped conv: {width} → {width} (32 groups, 4 channels each)")
    print(f"  3. 1×1 conv: {width} → 256 (expand)")
    print(f"  4. Residual connection")
    
    # Compare variants
    print("\n" + "=" * 70)
    print("ResNeXt Family")
    print("=" * 70)
    
    variants = {
        'ResNeXt-50 32×4d': resnext50_32x4d(),
        'ResNeXt-101 32×8d': resnext101_32x8d(),
        'ResNeXt-101 64×4d': resnext101_64x4d()
    }
    
    for name, model_variant in variants.items():
        params = sum(p.numel() for p in model_variant.parameters()) / 1e6
        # Extract configuration from name
        config = name.split()[-1]
        print(f"{name:25} {params:6.1f}M parameters  ({config})")
    
    # Grouped convolution explanation
    print("\n" + "=" * 70)
    print("Grouped Convolution Example (32 groups)")
    print("=" * 70)
    print("Standard convolution:")
    print("  - 128 input channels → 128 output channels")
    print("  - Each output sees ALL 128 inputs")
    print("  - Connections: 128 × 128 = 16,384")
    
    print("\nGrouped convolution (groups=32):")
    print("  - 128 inputs split into 32 groups (4 channels each)")
    print("  - 128 outputs split into 32 groups (4 channels each)")
    print("  - Each output group sees only its input group")
    print("  - Connections per group: 4 × 4 = 16")
    print("  - Total connections: 16 × 32 = 512")
    print("  - Reduction factor: 16,384 / 512 = 32×")
    
    # Cardinality vs Width
    print("\n" + "=" * 70)
    print("Cardinality vs Width (Similar FLOPs)")
    print("=" * 70)
    print("Configuration          Error   Insight")
    print("-" * 70)
    print("Width 64, C=1         22.2%   Baseline (ResNet)")
    print("Width 40, C=2         21.7%   Slight improvement")
    print("Width 28, C=4         21.4%   Better!")
    print("Width 14, C=8         21.3%   Even better!")
    print("Width 4,  C=32        21.2%   Best! (ResNeXt)")
    print("\nConclusion: Increasing cardinality > increasing width")
    
    # Performance comparison
    print("\n" + "=" * 70)
    print("Performance (ImageNet-1K)")
    print("=" * 70)
    print("Model               Params  FLOPs   Top-1   Notes")
    print("-" * 70)
    print("ResNet-50           25M     4.1G    76.1%   Baseline")
    print("ResNeXt-50 32×4d    25M     4.2G    77.6%   +1.5% same complexity")
    print("ResNeXt-50 32×4d    25M     4.2G    81.2%   With improved training")
    print()
    print("ResNet-101          44M     7.8G    77.4%   Deeper baseline")
    print("ResNeXt-101 32×8d   89M    16.4G    79.3%   2× params, +1.9%")
    print("ResNeXt-101 32×8d   89M    16.4G    82.8%   With improved training")
    
    # Key innovations
    print("\n" + "=" * 70)
    print("Key Innovations")
    print("=" * 70)
    print("✓ Cardinality: New dimension for scaling networks")
    print("✓ Grouped convolutions: Efficient multi-path implementation")
    print("✓ Homogeneous topology: All paths identical (easy to design)")
    print("✓ Split-transform-merge: Simple, modular strategy")
    print("✓ Better scaling: Higher cardinality > wider channels")
    
    # Historical impact
    print("\n" + "=" * 70)
    print("Historical Impact")
    print("=" * 70)
    print("Influenced:")
    print("  • ShuffleNet: Channel shuffle after grouped conv")
    print("  • MobileNet v2: Depthwise separable (extreme grouping)")
    print("  • Neural Architecture Search: Explore cardinality dimension")
    print("  • EfficientNet: Multi-dimension scaling (depth, width, resolution)")
    
    print("\n" + "=" * 70)

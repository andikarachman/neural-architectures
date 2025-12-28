"""
ConvNeXt: A ConvNet for the 2020s

ConvNeXt was introduced in "A ConvNet for the 2020s" by Liu et al. (Facebook AI 
Research / Meta AI, 2022).

THE PARADIGM SHIFT: MODERNIZING CONVNETS
=========================================
The 2010s were dominated by Vision Transformers (ViT):
- ViT showed CNNs might be obsolete
- Transformers achieved state-of-art on ImageNet
- Self-attention seemed superior to convolution
- CNNs appeared to have reached their limits

ConvNeXt's Challenge:
"Can a pure ConvNet compete with modern Vision Transformers?"

The Answer: YES!
- ConvNeXt matches or exceeds Swin Transformer performance
- Pure convolutional architecture (no self-attention)
- Simpler than transformers (no attention mechanism)
- Better scaling properties than traditional CNNs

Key Insight: "The performance gap between CNNs and transformers is due to 
             training recipes and architectural designs, not the fundamental 
             superiority of attention mechanisms."

MODERNIZATION ROADMAP
======================
ConvNeXt systematically modernizes ResNet by adopting design choices from 
Vision Transformers:

Starting Point: ResNet-50
Goal: Match Swin Transformer accuracy

Step-by-Step Improvements:
1. Training recipe → +2.7% accuracy
2. Macro design (stage ratio) → +0.4%
3. ResNeXt-ify (grouped conv) → +0.1%
4. Inverted bottleneck → +0.5%
5. Large kernel (7×7) → +0.8%
6. Micro design (ReLU→GELU, BN→LN) → +0.7%

Total improvement: ResNet-50 (78.8%) → ConvNeXt-T (82.1%)
Result: Matches Swin-T performance!

THE ARCHITECTURE: MODERNIZED CONVNET
=====================================

Traditional ResNet Block:
    Input (C)
       ↓
    1×1 Conv (C → C/4)  [Compress]
       ↓
    3×3 Conv (C/4 → C/4) [Spatial]
       ↓
    1×1 Conv (C/4 → C)  [Expand]
       ↓
    + [Residual]
       ↓
    Output (C)

ConvNeXt Block (Inverted Bottleneck):
    Input (C)
       ↓
    7×7 DWConv (C → C)  [Spatial, Depthwise]
       ↓
    LayerNorm
       ↓
    1×1 Conv (C → 4C)   [Expand first!]
       ↓
    GELU
       ↓
    1×1 Conv (4C → C)   [Compress back]
       ↓
    + [Residual]
       ↓
    Output (C)

Key Differences:
- Inverted bottleneck: expand then compress (like MobileNet)
- Large kernel: 7×7 instead of 3×3
- Depthwise convolution: spatial mixing per channel
- LayerNorm: instead of BatchNorm
- GELU: instead of ReLU
- Fewer activations: only one GELU (not three ReLUs)
- Fewer normalizations: only one LayerNorm

DESIGN PRINCIPLES
=================

1. TRAINING RECIPE (from Vision Transformers):
   - AdamW optimizer (not SGD with momentum)
   - Data augmentation: Mixup, Cutmix, RandAugment
   - Regularization: Stochastic Depth, Label Smoothing
   - Longer training: 300 epochs (not 90)
   
   Impact: +2.7% accuracy on ResNet-50
   Lesson: Training matters as much as architecture!

2. MACRO DESIGN (stage ratio):
   ResNet: (3, 4, 6, 3) blocks per stage
   Swin-T: (3, 3, 9, 3) blocks per stage
   ConvNeXt: (3, 3, 9, 3) [copy Swin]
   
   Why? More computation in later stages (richer features)
   Impact: +0.4% accuracy

3. PATCHIFY STEM:
   ResNet: 7×7 conv, stride=2 → pool
   ConvNeXt: 4×4 conv, stride=4
   
   Why? Matches ViT's non-overlapping patches
   - Aggressive downsampling early
   - More efficient (fewer FLOPs)

4. RESNEXT-IFY (grouped convolution):
   Standard conv: All channels interact
   Grouped conv: FLOPs reduced, accuracy similar
   Depthwise conv: Groups = channels (extreme)
   
   ConvNeXt uses depthwise convolution
   - Like MobileNet v2
   - Separates spatial and channel mixing
   Impact: +0.1% accuracy

5. INVERTED BOTTLENECK:
   Traditional: narrow → wide → narrow
   Inverted: wide → wider → wide
   
   ResNet: C → C/4 → C/4 → C
   ConvNeXt: C → 4C → C
   
   Why? Matches transformer MLP structure
   - Efficient: fewer memory-heavy 1×1 convs
   - Effective: more capacity in middle
   Impact: +0.5% accuracy

6. LARGE KERNELS (7×7):
   ResNet: 3×3 kernels everywhere
   Swin: Window size 7 (local attention range)
   ConvNeXt: 7×7 depthwise conv
   
   Why?
   - Larger receptive field
   - Global context (similar to attention)
   - Depthwise makes it efficient
   
   Progression tested: 3×3 → 5×5 → 7×7 → 9×9 → 11×11
   Sweet spot: 7×7 (9×9 and 11×11 marginally worse)
   Impact: +0.8% accuracy

7. MICRO DESIGN (activation & normalization):
   
   a) Fewer activations:
      ResNet: ReLU after every conv
      ConvNeXt: GELU only after expansion
      
      Why? Transformers use fewer activations
      Impact: Slight improvement
   
   b) GELU instead of ReLU:
      ReLU: max(0, x)
      GELU: x × Φ(x) [smoother, non-monotonic]
      
      Why? Standard in transformers (BERT, GPT)
      Impact: Marginal improvement
   
   c) LayerNorm instead of BatchNorm:
      BN: Normalize across batch dimension
      LN: Normalize across channel dimension
      
      Why?
      - Transformers use LayerNorm
      - No batch statistics (more stable)
      - Better for small batches
      
      Impact: +0.1% accuracy
   
   d) Separate downsampling layers:
      ResNet: Downsample in first block of stage
      ConvNeXt: Separate 2×2 conv, stride=2 layer
      
      Why? Cleaner, easier to understand
      Impact: Slight improvement
   
   Total micro design impact: +0.7% accuracy

CONVNEXT VARIANTS
=================
Four sizes, matching Swin Transformer compute:

ConvNeXt-T (Tiny):
- Channels: [96, 192, 384, 768]
- Depths: [3, 3, 9, 3]
- Parameters: 28M
- FLOPs: 4.5G
- Comparable: Swin-T, ResNet-50

ConvNeXt-S (Small):
- Channels: [96, 192, 384, 768]
- Depths: [3, 3, 27, 3]
- Parameters: 50M
- FLOPs: 8.7G
- Comparable: Swin-S, ResNet-101

ConvNeXt-B (Base):
- Channels: [128, 256, 512, 1024]
- Depths: [3, 3, 27, 3]
- Parameters: 89M
- FLOPs: 15.4G
- Comparable: Swin-B, ResNet-152

ConvNeXt-L (Large):
- Channels: [192, 384, 768, 1536]
- Depths: [3, 3, 27, 3]
- Parameters: 197M
- FLOPs: 34.4G
- Comparable: Swin-L

ConvNeXt-XL (Extra Large):
- Channels: [256, 512, 1024, 2048]
- Depths: [3, 3, 27, 3]
- Parameters: 350M
- FLOPs: 60.9G

LAYER SCALE
===========
Introduced to stabilize training of deep networks:

    y = x + γ · F(x)

Where:
- γ: Learnable scalar per channel (initialized to small value, e.g., 1e-6)
- F(x): Block transformation
- Allows smooth gradient flow initially
- Network learns to amplify useful transformations

Why it works:
- Initial small γ prevents instability in deep networks
- Gradually increases during training
- Each layer can learn its own importance
- Similar to fixup initialization

STOCHASTIC DEPTH
=================
Randomly drop entire residual blocks during training:

    y = x + DropPath(F(x), p)

Where:
- p: Drop probability (increases linearly with depth)
- Randomly skips entire blocks
- Test time: No dropping

Benefits:
- Implicit ensemble of networks
- Reduces overfitting
- Enables training very deep networks
- Similar to dropout but for entire layers

PERFORMANCE HIGHLIGHTS
======================
ImageNet-1K (224×224):
- ConvNeXt-T: 82.1% (vs Swin-T: 81.3%, ResNet-50: 78.8%)
- ConvNeXt-S: 83.1% (vs Swin-S: 83.0%, ResNet-101: 81.5%)
- ConvNeXt-B: 83.8% (vs Swin-B: 83.5%, ResNet-152: 82.8%)
- ConvNeXt-L: 84.3% (vs Swin-L: 84.5%)

Key Observations:
- Matches or exceeds Swin Transformer
- Simpler architecture (no attention)
- Better scaling: larger models improve more
- Efficient: competitive FLOPs

Transfer Learning:
- COCO object detection: 51.9 box mAP
- ADE20K segmentation: 53.7 mIoU
- Strong transfer across tasks

ADVANTAGES
==========
✓ Pure ConvNet (no attention mechanism)
✓ Matches transformer performance
✓ Simpler architecture than ViT/Swin
✓ Efficient: good accuracy/FLOPs ratio
✓ Good scaling properties
✓ Excellent transfer learning
✓ Easier to understand and modify
✓ Isotropic design (mostly uniform layers)

LIMITATIONS
===========
✗ Still slower than some transformers at inference
✗ Large kernel size increases memory usage
✗ Requires modern training recipe for full performance
✗Depthwise convolutions not optimized on all hardware
✗ LayerNorm slower than BatchNorm on some devices

KEY TAKEAWAYS
=============
1. Architecture alone doesn't explain transformer success
2. Training recipe matters enormously (+2.7% accuracy)
3. Pure ConvNets can match modern transformers
4. Design choices from transformers apply to CNNs
5. Inductive biases (translation equivariance) still valuable
6. Simple, clean architectures are competitive
7. Systematic modernization is effective

Historical Impact:
- Revived interest in convolutional architectures
- Showed transformers aren't always necessary
- Influenced: ConvNeXt v2, FastViT, RepLKNet
- Demonstrated importance of training methodology

Reference:
    Zhuang Liu et al. "A ConvNet for the 2020s." 
    CVPR 2022. arXiv:2201.03545
    https://arxiv.org/abs/2201.03545
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from functools import partial


class LayerNorm(nn.Module):
    """
    LayerNorm that can handle both (B, C, H, W) and (B, H, W, C) formats.
    
    Standard nn.LayerNorm expects (B, H, W, C), but CNNs traditionally use (B, C, H, W).
    ConvNeXt uses (B, H, W, C) format inside blocks for compatibility with nn.Linear.
    
    Args:
        normalized_shape: Number of channels to normalize
        eps: Epsilon for numerical stability
        data_format: Either "channels_last" or "channels_first"
    """
    def __init__(
        self, 
        normalized_shape: int, 
        eps: float = 1e-6, 
        data_format: str = "channels_last"
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) if channels_first, (B, H, W, C) if channels_last
        Returns:
            Same format as input, normalized tensor
        """
        if self.data_format == "channels_last":
            # (B, H, W, C): Use standard LayerNorm
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        else:
            # (B, C, H, W): Normalize over channel dimension
            u = x.mean(1, keepdim=True)  # (B, 1, H, W)
            s = (x - u).pow(2).mean(1, keepdim=True)  # (B, 1, H, W)
            x = (x - u) / torch.sqrt(s + self.eps)  # (B, C, H, W)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DropPath(nn.Module):
    """
    Stochastic Depth: Drop entire residual paths with given probability.
    
    During training, randomly drops the entire transformation with probability p.
    At test time, scales the output by (1 - p) to compensate.
    
    This is different from Dropout:
    - Dropout: Randomly zeros individual activations
    - DropPath: Randomly drops entire residual branch
    
    Args:
        drop_prob: Probability of dropping path
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        # Binary mask: 1 = keep, 0 = drop
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        
        # Scale by keep_prob during training
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block: Modernized inverted bottleneck.
    
    Architecture (reading order matches forward pass):
    1. 7×7 Depthwise Conv (spatial mixing per channel)
    2. LayerNorm (normalize features)
    3. 1×1 Conv (expand: C → 4C)
    4. GELU activation
    5. 1×1 Conv (compress: 4C → C)
    6. Layer Scale (learnable per-channel scaling)
    7. Drop Path (stochastic depth)
    8. Residual connection
    
    Key Design:
    - Inverted bottleneck: expand first, then compress
    - Large kernel (7×7) for bigger receptive field
    - Depthwise conv for efficiency
    - Single activation (not three like ResNet)
    - LayerNorm instead of BatchNorm
    - GELU instead of ReLU
    
    Args:
        dim: Number of input/output channels
        drop_path: Stochastic depth probability
        layer_scale_init_value: Initial value for layer scale (1e-6 in paper)
    """
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        
        # 1. Depthwise convolution (7×7)
        # groups=dim means each channel has its own filter
        # This does spatial mixing within each channel independently
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=7,
            padding=3,
            groups=dim  # Depthwise: each channel processed separately
        )
        
        # 2. LayerNorm (channels_last format after permutation)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        
        # 3. Pointwise/1×1 convolution: expansion (C → 4C)
        # This does channel mixing (no spatial mixing)
        # 4× expansion matches transformer MLP ratio
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        
        # 4. Activation function
        # GELU is smoother than ReLU, standard in transformers
        self.act = nn.GELU()
        
        # 5. Pointwise/1×1 convolution: compression (4C → C)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # 6. Layer Scale: per-channel learnable scaling
        # Initialized to small value (1e-6) for training stability
        # Allows network to learn importance of each layer
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(dim),
                requires_grad=True
            )
        else:
            self.gamma = None
        
        # 7. Stochastic Depth (Drop Path)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
        Returns:
            (B, C, H, W) output tensor
        """
        input = x
        
        # 1. Depthwise convolution (spatial mixing)
        x = self.dwconv(x)  # (B, C, H, W) → (B, C, H, W)
        
        # 2. Permute for LayerNorm: (B, C, H, W) → (B, H, W, C)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        # 3. LayerNorm
        x = self.norm(x)  # (B, H, W, C)
        
        # 4. First pointwise conv (expansion)
        # Note: pwconv is actually nn.Linear, operates on last dim
        x = self.pwconv1(x)  # (B, H, W, C) → (B, H, W, 4C)
        
        # 5. GELU activation
        x = self.act(x)  # (B, H, W, 4C)
        
        # 6. Second pointwise conv (compression)
        x = self.pwconv2(x)  # (B, H, W, 4C) → (B, H, W, C)
        
        # 7. Layer Scale (if enabled)
        if self.gamma is not None:
            x = self.gamma * x  # Element-wise scaling
        
        # 8. Permute back: (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # 9. Drop Path (stochastic depth)
        x = self.drop_path(x)
        
        # 10. Residual connection
        x = input + x
        
        return x


class ConvNeXt(nn.Module):
    """
    ConvNeXt: A ConvNet for the 2020s.
    
    Pure convolutional network that matches Vision Transformer performance.
    Systematically modernizes ResNet with design choices from transformers.
    
    Architecture:
    1. Stem: Aggressive downsampling (4×4 conv, stride=4)
       - Like ViT's patchify operation
       - Reduces resolution 4×
    
    2. Four stages with progressively:
       - Increasing channels: [C, 2C, 4C, 8C]
       - Increasing depth: [3, 3, 9/27, 3] blocks
       - Decreasing resolution: /2 per stage
    
    3. Each stage:
       - Downsampling layer (2×2 conv, stride=2) [except first]
       - Multiple ConvNeXt blocks
       - Stochastic depth (linearly increasing)
    
    4. Head:
       - Global average pooling
       - Layer norm
       - Linear classifier
    
    ConvNeXt Family (C = base channels, D = depths):
    - ConvNeXt-T: C=[96, 192, 384, 768], D=[3, 3, 9, 3], 28M params
    - ConvNeXt-S: C=[96, 192, 384, 768], D=[3, 3, 27, 3], 50M params
    - ConvNeXt-B: C=[128, 256, 512, 1024], D=[3, 3, 27, 3], 89M params
    - ConvNeXt-L: C=[192, 384, 768, 1536], D=[3, 3, 27, 3], 197M params
    
    Args:
        in_chans: Number of input channels (3 for RGB)
        num_classes: Number of output classes
        depths: Number of blocks per stage
        dims: Number of channels per stage
        drop_path_rate: Maximum drop path rate (linearly increases)
        layer_scale_init_value: Initial layer scale value
        head_init_scale: Scaling factor for head initialization
    """
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0
    ):
        super().__init__()
        
        # ========== STEM ==========
        # Aggressive downsampling: 4×4 conv with stride 4
        # Mimics ViT's patch embedding (non-overlapping patches)
        # Input: (B, 3, 224, 224) → Output: (B, 96, 56, 56)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # ========== DOWNSAMPLING LAYERS ==========
        # Between stages: 2×2 conv with stride 2
        # Also increases channels: C → 2C
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)
        
        # ========== STAGES ==========
        # Four stages of ConvNeXt blocks
        # Stochastic depth rate increases linearly with depth
        self.stages = nn.ModuleList()
        
        # Calculate drop path rates (linearly increasing)
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        
        cur = 0  # Current block index (for drop path)
        for i in range(4):
            # Create stage with multiple blocks
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # ========== CLASSIFICATION HEAD ==========
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # Final norm
        self.head = nn.Linear(dims[-1], num_classes)  # Classifier
        
        # Initialize weights
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        
    def _init_weights(self, m):
        """
        Initialize weights following ConvNeXt paper.
        
        - Conv2d and Linear: Truncated normal (std=0.02)
        - LayerNorm: weight=1, bias=0
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features (before classification head).
        
        Useful for:
        - Transfer learning
        - Feature extraction
        - Dense prediction tasks
        
        Args:
            x: (B, 3, H, W) input images
        Returns:
            (B, dims[-1], H/32, W/32) features
        """
        for i in range(4):
            x = self.downsample_layers[i](x)  # Downsample
            x = self.stages[i](x)              # Process through blocks
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            x: (B, 3, H, W) input images (typically 224×224)
        Returns:
            (B, num_classes) logits
        """
        # Extract features through stages
        x = self.forward_features(x)  # (B, dims[-1], H/32, W/32)
        
        # Global average pooling
        x = x.mean([-2, -1])  # (B, dims[-1])
        
        # Layer norm
        x = self.norm(x)  # (B, dims[-1])
        
        # Classification head
        x = self.head(x)  # (B, num_classes)
        
        return x


# ========== MODEL VARIANTS ==========

def convnext_tiny(num_classes: int = 1000, **kwargs):
    """
    ConvNeXt-Tiny (ConvNeXt-T)
    
    Configuration:
    - Channels: [96, 192, 384, 768]
    - Depths: [3, 3, 9, 3]
    - Parameters: ~28M
    - FLOPs: ~4.5G
    
    Comparable to:
    - Swin-T (28M params)
    - ResNet-50 (25M params)
    
    Performance:
    - ImageNet-1K: 82.1% top-1 accuracy
    """
    model = ConvNeXt(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        num_classes=num_classes,
        **kwargs
    )
    return model


def convnext_small(num_classes: int = 1000, **kwargs):
    """
    ConvNeXt-Small (ConvNeXt-S)
    
    Configuration:
    - Channels: [96, 192, 384, 768]
    - Depths: [3, 3, 27, 3]
    - Parameters: ~50M
    - FLOPs: ~8.7G
    
    Comparable to:
    - Swin-S (50M params)
    - ResNet-101 (44M params)
    
    Performance:
    - ImageNet-1K: 83.1% top-1 accuracy
    """
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        num_classes=num_classes,
        **kwargs
    )
    return model


def convnext_base(num_classes: int = 1000, **kwargs):
    """
    ConvNeXt-Base (ConvNeXt-B)
    
    Configuration:
    - Channels: [128, 256, 512, 1024]
    - Depths: [3, 3, 27, 3]
    - Parameters: ~89M
    - FLOPs: ~15.4G
    
    Comparable to:
    - Swin-B (88M params)
    - ResNet-152 (60M params)
    
    Performance:
    - ImageNet-1K: 83.8% top-1 accuracy
    - ImageNet-22K pre-train: 85.8% top-1 accuracy
    """
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        num_classes=num_classes,
        **kwargs
    )
    return model


def convnext_large(num_classes: int = 1000, **kwargs):
    """
    ConvNeXt-Large (ConvNeXt-L)
    
    Configuration:
    - Channels: [192, 384, 768, 1536]
    - Depths: [3, 3, 27, 3]
    - Parameters: ~197M
    - FLOPs: ~34.4G
    
    Comparable to:
    - Swin-L (197M params)
    
    Performance:
    - ImageNet-1K: 84.3% top-1 accuracy
    - ImageNet-22K pre-train: 86.6% top-1 accuracy
    """
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        num_classes=num_classes,
        **kwargs
    )
    return model


def convnext_xlarge(num_classes: int = 1000, **kwargs):
    """
    ConvNeXt-XLarge (ConvNeXt-XL)
    
    Configuration:
    - Channels: [256, 512, 1024, 2048]
    - Depths: [3, 3, 27, 3]
    - Parameters: ~350M
    - FLOPs: ~60.9G
    
    Performance:
    - ImageNet-22K pre-train: 87.0% top-1 accuracy (on ImageNet-1K)
    
    Note: Only trained with ImageNet-22K pre-training
    """
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[256, 512, 1024, 2048],
        num_classes=num_classes,
        **kwargs
    )
    return model


if __name__ == "__main__":
    # ========== DEMONSTRATION ==========
    print("=" * 70)
    print("ConvNeXt: A ConvNet for the 2020s")
    print("=" * 70)
    
    # Create ConvNeXt-Tiny model
    model = convnext_tiny(num_classes=1000)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nConvNeXt-Tiny Architecture:")
    print(f"  Depths: [3, 3, 9, 3]")
    print(f"  Channels: [96, 192, 384, 768]")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Example input (ImageNet size)
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output logits sample: {output[0, :5]}")
    
    # Show feature extraction
    print("\n" + "=" * 70)
    print("Feature Extraction")
    print("=" * 70)
    
    with torch.no_grad():
        features = model.forward_features(x)
    
    print(f"Feature shape: {features.shape}")
    print(f"Spatial reduction: 224 → {features.shape[2]} (32×)")
    
    # Compare all variants
    print("\n" + "=" * 70)
    print("ConvNeXt Family")
    print("=" * 70)
    
    variants = {
        'ConvNeXt-T': convnext_tiny(),
        'ConvNeXt-S': convnext_small(),
        'ConvNeXt-B': convnext_base(),
        'ConvNeXt-L': convnext_large(),
        'ConvNeXt-XL': convnext_xlarge()
    }
    
    for name, model in variants.items():
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{name:15} {params:6.1f}M parameters")
    
    # Show block structure
    print("\n" + "=" * 70)
    print("ConvNeXt Block Structure")
    print("=" * 70)
    
    block = ConvNeXtBlock(dim=96)
    x_block = torch.randn(1, 96, 56, 56)
    
    print(f"Input:  {x_block.shape}")
    with torch.no_grad():
        y_block = block(x_block)
    print(f"Output: {y_block.shape}")
    
    print("\nBlock components:")
    print("  1. 7×7 Depthwise Conv (spatial mixing)")
    print("  2. LayerNorm")
    print("  3. 1×1 Conv (expand: C → 4C)")
    print("  4. GELU activation")
    print("  5. 1×1 Conv (compress: 4C → C)")
    print("  6. Layer Scale")
    print("  7. Drop Path")
    print("  8. Residual connection")
    
    # Key innovations
    print("\n" + "=" * 70)
    print("Key Innovations")
    print("=" * 70)
    print("✓ Inverted bottleneck (expand first, like MobileNet)")
    print("✓ Large kernels (7×7 instead of 3×3)")
    print("✓ Depthwise convolution (efficient spatial mixing)")
    print("✓ LayerNorm instead of BatchNorm")
    print("✓ GELU instead of ReLU")
    print("✓ Fewer activations (only one per block)")
    print("✓ Layer Scale (training stability)")
    print("✓ Stochastic Depth (regularization)")
    
    # Performance comparison
    print("\n" + "=" * 70)
    print("Performance (ImageNet-1K)")
    print("=" * 70)
    print("Model          Params  FLOPs   Top-1")
    print("-" * 44)
    print("ResNet-50      25M     4.1G    78.8%")
    print("Swin-T         28M     4.5G    81.3%")
    print("ConvNeXt-T     28M     4.5G    82.1%  ← Best!")
    print()
    print("ResNet-101     44M     7.8G    81.5%")
    print("Swin-S         50M     8.7G    83.0%")
    print("ConvNeXt-S     50M     8.7G    83.1%  ← Best!")
    print()
    print("ResNet-152     60M    11.3G    82.8%")
    print("Swin-B         88M    15.4G    83.5%")
    print("ConvNeXt-B     89M    15.4G    83.8%  ← Best!")
    
    print("\n" + "=" * 70)
    print("Comparison: ConvNeXt vs Vision Transformers")
    print("=" * 70)
    print("ConvNeXt:")
    print("  ✓ Pure convolution (no attention)")
    print("  ✓ Simpler architecture")
    print("  ✓ Translation equivariance (inductive bias)")
    print("  ✓ Competitive or better accuracy")
    print("  ✓ Good transfer learning")
    
    print("\nVision Transformers:")
    print("  ✓ Global receptive field (attention)")
    print("  ✓ Flexible architecture")
    print("  ✓ Better scaling to huge datasets")
    print("  ✓ Strong few-shot learning")
    
    print("\n" + "=" * 70)

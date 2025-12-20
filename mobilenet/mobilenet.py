"""
MobileNet V1: Efficient Convolutional Neural Network (Didactic Version)

MobileNet V1 was introduced in:
"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
— Howard et al. (2017)

THE CORE INNOVATION: DEPTHWISE SEPARABLE CONVOLUTION
===================================================
Standard convolution is expensive because it does TWO things at once:

  (1) Spatial filtering (look around in H×W using a K×K kernel)
  (2) Channel mixing   (combine information across input channels)

MobileNet V1 factorizes this into TWO cheaper steps:

  A) Depthwise Convolution (DW):
     - One K×K filter PER input channel (no channel mixing)
     - Performs spatial filtering cheaply

  B) Pointwise Convolution (PW):
     - A 1×1 convolution that mixes channels
     - Performs channel mixing cheaply

So instead of:
  StandardConv(K×K):  (C_in -> C_out)

We do:
  DepthwiseConv(K×K): (C_in -> C_in)   [groups = C_in]
  PointwiseConv(1×1): (C_in -> C_out)

WHY THIS SAVES COMPUTE (INTUITION + SIMPLE MATH)
================================================
Let:
  - input feature map size be H×W
  - kernel size be K×K (usually 3×3)
  - input channels C_in
  - output channels C_out

Standard conv FLOPs (roughly):
  H*W * (K*K * C_in) * C_out

Depthwise separable FLOPs:
  Depthwise: H*W * (K*K * C_in)
  Pointwise: H*W * (C_in * C_out)
  Total    : H*W * (K*K*C_in + C_in*C_out)

Savings ratio (approx):
  (K*K*C_in*C_out) / (K*K*C_in + C_in*C_out)
= (K*K*C_out) / (K*K + C_out)

Example (K=3, C_out=256):
  ratio ≈ (9*256) / (9+256) ≈ 2304 / 265 ≈ 8.7× cheaper

This is why MobileNet is so fast.

MOBILENET V1 ARCHITECTURE (ImageNet-style, 224×224)
===================================================
MobileNet V1 uses a simple pattern:

  Conv3×3(stride=2)  -> DW+PW blocks with occasional stride=2 downsampling
  ... -> GlobalAvgPool -> FC

Each DW+PW block:
  DepthwiseConv3×3(groups=C_in) -> BN -> ReLU
  PointwiseConv1×1              -> BN -> ReLU
"""

import torch
import torch.nn as nn

def conv3x3(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    """
    Standard 3x3 convolution block:
        Conv2d -> BatchNorm2d -> ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3,
                  stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution (MobileNet V1 block)

    Step A) Depthwise 3×3:
      - groups = in_channels
      - each channel has its own 3×3 filter
      - NO channel mixing here

    Step B) Pointwise 1×1:
      - mixes channels
      - changes channel count (in_channels -> out_channels)

    Block structure:
      DW Conv3×3 -> BN -> ReLU -> PW Conv1×1 -> BN -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()

        # Depthwise convolution: spatial filtering only
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Pointwise convolution: channel mixing only
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    

class MobileNetV1(nn.Module):
    """
    MobileNet V1 (ImageNet-style).

    Downsampling schedule (224 -> 112 -> 56 -> 28 -> 14 -> 7):
      - Initial conv: stride=2 (224 -> 112)
      - Some DW blocks have stride=2 to downsample further

    Channel progression:
      32, 64, 128, 256, 512, 1024

    Note:
      MobileNet V1 has "width multiplier" alpha in the paper, but
      we omit it here for clarity. This is the canonical alpha=1.0 version.
    """
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()

        # Input: (N, 3, 224, 224)

        # Stem: Initial Standard Conv Layer
        self.stem = conv3x3(3, 32, stride=2)  # (N, 32, 112, 112)

        # MobileNet V1 Blocks (canonical configuration)
        # Format: (in_channels, out_channels, stride)
        blocks_cfg = [
            (32,   64,   1),  # (N, 64, 112, 112)
            (64,   128,  2),  # (N, 128, 56, 56)
            (128,  128,  1),  # (N, 128, 56, 56)
            (128,  256,  2),  # (N, 256, 28, 28)
            (256,  256,  1),  # (N, 256, 28, 28)
            (256,  512,  2),  # (N, 512, 14, 14)
            # 5 blocks at 512 channels (14x14)
            (512,  512,  1),  # (N, 512, 14, 14)
            (512,  512,  1),  # (N, 512, 14, 14)
            (512,  512,  1),  # (N, 512, 14, 14)
            (512,  512,  1),  # (N, 512, 14, 14)
            (512,  512,  1),  # (N, 512, 14, 14)
            (512,  1024, 2),  # (N, 1024, 7, 7)
            (1024, 1024, 1),  # (N, 1024, 7, 7)
        ]

        layers = []
        for in_c, out_c, stride in blocks_cfg:
            layers.append(DepthwiseSeparableConv(in_c, out_c, stride))
        
        self.blocks = nn.Sequential(*layers)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # (N, 1024, 1, 1)
        self.fc = nn.Linear(1024, num_classes)       # (N, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)         # (N, 32, 112, 112)
        x = self.blocks(x)       # (N, 1024, 7, 7)
        x = self.avgpool(x)      # (N, 1024, 1, 1)
        x = torch.flatten(x, 1)  # Flatten to (N, 1024)
        x = self.fc(x)           # (N, num_classes)
        return x


if __name__ == "__main__":
    print("=" * 70)
    print("MobileNet V1 (ImageNet-style) — Didactic Sanity Check")
    print("=" * 70)

    model = MobileNetV1(num_classes=1000)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    print("\n[1] Input/Output")
    print("    input :", tuple(x.shape))
    print("    output:", tuple(y.shape))

    print("\n[2] Parameter count")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    total params: {total_params:,}")

    print("\n[3] Parameter breakdown (rough)")
    stem_params = sum(p.numel() for p in model.stem.parameters())
    feat_params = sum(p.numel() for p in model.blocks.parameters())
    head_params = sum(p.numel() for p in model.fc.parameters())

    print(f"    stem     : {stem_params:,}")
    print(f"    features : {feat_params:,}")
    print(f"    head(fc) : {head_params:,}")

    print("\n" + "=" * 70)
    print("Key takeaway: factorize conv into (depthwise spatial) + (pointwise channel mixing).")
    print("=" * 70)


"""
ResNet-18: Deep Residual Network Architecture

ResNet (Residual Network) was introduced in "Deep Residual Learning for Image Recognition"
by He et al. (2015) - Winner of ILSVRC 2015 with 3.57% error rate.

THE CORE INNOVATION: RESIDUAL LEARNING
========================================
Problem: Deep networks are hard to train due to vanishing/exploding gradients.
Solution: Instead of learning H(x) directly, learn the residual F(x) = H(x) - x
          Then: H(x) = F(x) + x

WHY RESIDUAL CONNECTIONS WORK:
==============================
1. GRADIENT FLOW: Skip connections provide direct paths for gradients to flow backward
   - Gradients can bypass layers, avoiding vanishing gradient problem
   - Enables training of very deep networks (50, 101, 152+ layers)

2. IDENTITY MAPPING: If optimal function is close to identity, easier to learn F(x) ≈ 0
   than to learn H(x) = x from scratch
   
3. ENSEMBLE EFFECT: Network can be viewed as ensemble of shorter paths
   - Each skip connection creates a different path length
   - Provides implicit model averaging

4. FEATURE REUSE: Earlier features can be directly used by later layers
   - Low-level features (edges, textures) remain accessible
   - Higher layers can refine or use them as-is

RESNET-18 ARCHITECTURE:
======================
Input (224x224x3)
  ↓
Stem: Conv7x7 → BN → ReLU → MaxPool (outputs 56x56x64)
  ↓
Stage 1: 2 BasicBlocks, 64 channels  (56x56x64)  - No downsampling
Stage 2: 2 BasicBlocks, 128 channels (28x28x128) - Downsample with stride=2
Stage 3: 2 BasicBlocks, 256 channels (14x14x256) - Downsample with stride=2  
Stage 4: 2 BasicBlocks, 512 channels (7x7x512)   - Downsample with stride=2
  ↓
Global Average Pooling (1x1x512)
  ↓
Fully Connected (num_classes)

Total: 18 weight layers (1 stem conv + 16 stage convs + 1 fc)

KEY DIFFERENCES FROM VGG:
=========================
1. Skip connections (residual blocks) - THE KEY INNOVATION
2. Batch Normalization used throughout (not optional)
3. No dropout in main architecture (BN provides regularization)
4. Global Average Pooling instead of large FC layers (fewer parameters)
5. Much deeper but fewer parameters than VGG (11M vs 138M for VGG16)
6. No bias in conv layers when followed by BN (BN has its own bias term)
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    BasicBlock: Fundamental building block for ResNet-18 and ResNet-34.
    
    RESIDUAL LEARNING FORMULA:
        Output = F(x) + x
        Where F(x) is the learned residual function (two conv layers)
        And x is the shortcut/skip connection (identity or projection)
    
    ARCHITECTURE:
        Main Path F(x):    x → Conv3x3 → BN → ReLU → Conv3x3 → BN
        Shortcut Path:     x → [Identity or Conv1x1] 
        Addition:          F(x) + shortcut
        Final Activation:  ReLU(F(x) + x)
    
    WHY TWO 3x3 CONVOLUTIONS?
    -------------------------
    - Two 3x3 convs have effective receptive field of 5x5
    - More non-linearity (two ReLU) than single larger kernel
    - Follows VGG philosophy but adds skip connections
    
    WHEN IS SHORTCUT A PROJECTION?
    ------------------------------
    When spatial dimensions or channels change:
    1. stride != 1: Spatial downsampling (H,W changes)
    2. in_channels != out_channels: Channel count changes
    
    Solution: Use 1x1 convolution to project x to match F(x) dimensions
    
    EXPANSION FACTOR:
    ----------------
    - BasicBlock.expansion = 1 (output channels = out_channels)
    - Used by ResNet-18/34
    - Bottleneck blocks (ResNet-50+) use expansion = 4
    """
    expansion = 1  # Output channels multiplier (1 for BasicBlock, 4 for Bottleneck)

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """
        Initialize BasicBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first conv layer
                   - stride=1: Maintains spatial dimensions (used for most blocks)
                   - stride=2: Downsamples by 2x (used at start of stages 2,3,4)
        """
        super().__init__()

        # ========== MAIN PATH: F(x) - Learn the Residual ==========
        
        # First Convolution: May downsample if stride > 1
        # - bias=False: Not needed because BatchNorm has bias parameter
        # - padding=1: Maintains spatial size when stride=1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second Convolution: Always stride=1 (no downsampling)
        # - Refines features at the current spatial resolution
        # - Output will be added to shortcut
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ========== SHORTCUT PATH: Identity or Projection ==========
        # Purpose: Match dimensions between input x and main path output F(x)
        
        self.shortcut = nn.Identity()  # Default: x passes through unchanged
        
        # Need projection when dimensions mismatch:
        if stride != 1 or in_channels != out_channels:
            # Use 1x1 convolution to:
            # 1. Change number of channels (in_channels → out_channels)
            # 2. Downsample spatially if stride > 1
            # Note: This is "projection shortcut" or "Conv shortcut"
            # Alternative approach: zero-padding (not used here)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BasicBlock implementing: H(x) = F(x) + x
        
        Key Insight:
        -----------
        The network learns to predict the RESIDUAL F(x) rather than the
        full transformation H(x). This is easier because:
        - If identity mapping is optimal, F(x) just needs to learn ~0
        - Gradients flow directly through skip connection
        
        Args:
            x: Input tensor (N, in_channels, H, W)
            
        Returns:
            Output tensor (N, out_channels, H/stride, W/stride)
        """
        # ========== SHORTCUT PATH: Compute identity/projection ==========
        # Save the input for residual connection
        # If dimensions mismatch, this applies 1x1 conv + BN
        shortcut = self.shortcut(x)  # (N, out_channels, H/stride, W/stride)
        
        # ========== MAIN PATH: Compute F(x) ==========
        # First conv block: Conv → BN → ReLU
        out = self.conv1(x)          # Apply first convolution (may downsample)
        out = self.bn1(out)          # Normalize activations
        out = self.relu(out)         # Non-linearity
        
        # Second conv block: Conv → BN (NO ReLU YET!)
        out = self.conv2(out)        # Apply second convolution
        out = self.bn2(out)          # Normalize activations
        # Note: ReLU is applied AFTER addition
        
        # ========== RESIDUAL CONNECTION: The Heart of ResNet ==========
        # Add the residual: H(x) = F(x) + x
        # This is where the magic happens!
        # - Enables gradient flow through skip connection
        # - Allows network to learn identity mapping if optimal
        # - Creates implicit ensemble of different depth paths
        out += shortcut              # Element-wise addition
        
        # Final activation: Apply ReLU after addition
        # This is called "post-activation" design
        out = self.relu(out)
        
        return out
    

class ResNet18(nn.Module):
    """
    ResNet-18: 18-layer Deep Residual Network.
    
    ARCHITECTURE BREAKDOWN:
    ======================
    Layer Name    | Output Size | Layers                                    | Params
    --------------|-------------|-------------------------------------------|--------
    Input         | 224×224×3   | -                                         | -
    Stem (conv1)  | 56×56×64    | 7×7 conv, stride 2 → BN → ReLU → MaxPool | ~9K
    Stage 1       | 56×56×64    | [3×3 conv] × 2, ×2 blocks                | ~74K
    Stage 2       | 28×28×128   | [3×3 conv] × 2, ×2 blocks, stride 2      | ~230K
    Stage 3       | 14×14×256   | [3×3 conv] × 2, ×2 blocks, stride 2      | ~920K
    Stage 4       | 7×7×512     | [3×3 conv] × 2, ×2 blocks, stride 2      | ~3.7M
    Classifier    | 1×1×512     | Global AvgPool → FC                       | ~512K
    Total         | -           | -                                         | ~11.7M
    
    STAGE STRUCTURE:
    ===============
    Each stage consists of multiple BasicBlocks:
    - Stage 1: 2 blocks, 64 channels,  stride=1 (no downsampling)
    - Stage 2: 2 blocks, 128 channels, stride=2 (first block downsamples)
    - Stage 3: 2 blocks, 256 channels, stride=2 (first block downsamples)
    - Stage 4: 2 blocks, 512 channels, stride=2 (first block downsamples)
    
    COMPARISON WITH VGG-16:
    ======================
    ResNet-18: ~11.7M parameters, 18 layers, uses skip connections
    VGG-16:    ~138M parameters, 16 layers, no skip connections
    
    ResNet is 12x smaller but typically more accurate due to:
    - Residual connections enabling deeper effective depth
    - Global average pooling (no huge FC layers)
    - Better gradient flow
    
    DESIGN CHOICES:
    ==============
    1. No bias in conv layers (BN handles bias)
    2. Batch Normalization after every convolution
    3. Global Average Pooling instead of flatten + FC layers
    4. Downsampling via stride=2 in conv layers (not pooling)
    5. Bottleneck design for deeper variants (ResNet-50+)
    """
    
    def __init__(self, num_classes: int = 1000) -> None:
        """
        Initialize ResNet-18.
        
        Args:
            num_classes: Number of output classes (default: 1000 for ImageNet)
        """
        super().__init__()

        # ========== STEM: Initial Feature Extraction ==========
        # Purpose: Quickly reduce spatial dimensions and extract low-level features
        # Spatial transformation: 224×224 → 112×112 → 56×56
        self.stem = nn.Sequential(
            # Large 7×7 convolution with stride 2
            # - Captures more context than 3×3 (larger receptive field)
            # - Stride 2 reduces spatial dimensions: 224×224 → 112×112
            # - padding=3 maintains proper spatial size
            # - bias=False because BatchNorm adds bias
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Max pooling for further downsampling: 112×112 → 56×56
            # - Provides translation invariance
            # - Reduces computational cost for subsequent layers
            # - stride=2, kernel=3, padding=1 is standard ResNet configuration
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (N, 64, 56, 56)
        )

        # ========== STAGE 1: First Residual Stage ==========
        # - Maintains spatial resolution (56×56)
        # - Learns refined features from stem output
        # - Both blocks use stride=1 (no downsampling)
        # - Pattern: [BasicBlock(stride=1), BasicBlock(stride=1)]
        self.stage1 = self._make_stage(
            in_channels=64, out_channels=64, num_blocks=2, first_stride=1
        )  # Output: (N, 64, 56, 56)
        
        # ========== STAGE 2: First Downsampling Stage ==========
        # - Doubles channels: 64 → 128
        # - Halves spatial size: 56×56 → 28×28 (first block with stride=2)
        # - Pattern: [BasicBlock(stride=2), BasicBlock(stride=1)]
        self.stage2 = self._make_stage(
            in_channels=64, out_channels=128, num_blocks=2, first_stride=2
        )  # Output: (N, 128, 28, 28)
        
        # ========== STAGE 3: Second Downsampling Stage ==========
        # - Doubles channels: 128 → 256
        # - Halves spatial size: 28×28 → 14×14
        # - Learns mid-to-high level features (object parts, patterns)
        self.stage3 = self._make_stage(
            in_channels=128, out_channels=256, num_blocks=2, first_stride=2
        )  # Output: (N, 256, 14, 14)
        
        # ========== STAGE 4: Final Feature Stage ==========
        # - Doubles channels: 256 → 512
        # - Halves spatial size: 14×14 → 7×7
        # - Learns high-level semantic features (complete objects, concepts)
        self.stage4 = self._make_stage(
            in_channels=256, out_channels=512, num_blocks=2, first_stride=2
        )  # Output: (N, 512, 7, 7)

        # ========== CLASSIFICATION HEAD ==========
        
        # Global Average Pooling: Aggregates spatial information
        # - Converts (N, 512, 7, 7) → (N, 512, 1, 1)
        # - Averages each 7×7 feature map to a single value
        # - Benefits: Position invariant, no parameters, reduces overfitting
        # - Much better than flatten + large FC layers (VGG approach)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final fully connected layer
        # - Maps 512 features to class predictions
        # - Only ~512K parameters (vs ~100M in VGG's FC layers)
        # - expansion factor allows same code for ResNet-50+ (expansion=4)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    

    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, first_stride: int) -> nn.Sequential:
        """
        Construct a ResNet stage consisting of multiple BasicBlocks.
        
        A stage is a sequence of residual blocks at the same spatial resolution
        (except the first block may downsample).
        
        Args:
            in_channels: Input channels for the first block
            out_channels: Output channels for all blocks in this stage
            num_blocks: Number of BasicBlocks in this stage (2 for ResNet-18)
            first_stride: Stride for the first block
                         - stride=1: Maintains spatial dimensions (Stage 1)
                         - stride=2: Downsamples by 2× (Stages 2, 3, 4)
        
        Returns:
            nn.Sequential containing the stage's BasicBlocks
        
        Stage Pattern:
        -------------
        Block 1: May change channels and/or spatial size (uses first_stride)
                 - If stride=2: Downsampling occurs here
                 - If in_channels != out_channels: Channel adjustment occurs
                 - Shortcut uses projection (1×1 conv)
        
        Blocks 2-N: Maintain same channels and spatial size (stride=1)
                   - Process features at current resolution
                   - Shortcuts use identity mapping (no conv needed)
        
        Example for Stage 2 (first_stride=2):
            Block 1: (N, 64, 56, 56)  → (N, 128, 28, 28)  [stride=2, projection shortcut]
            Block 2: (N, 128, 28, 28) → (N, 128, 28, 28)  [stride=1, identity shortcut]
        """
        blocks = []
        
        # First block: May downsample and/or change channels
        # This is where dimension transformations happen
        blocks.append(BasicBlock(in_channels, out_channels, stride=first_stride))
        
        # Remaining blocks: Maintain dimensions
        # These refine features at the current spatial resolution
        # All use identity shortcuts (no projection needed)
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet-18.
        
        Data flows through progressively deeper feature representations:
        1. Stem: Rapid downsampling and initial feature extraction
        2. Stage 1-4: Hierarchical feature learning with residual connections
        3. Global pooling: Spatial aggregation
        4. Classifier: Final predictions
        
        Key Insight:
        -----------
        At each stage, the network can:
        - Learn new transformations via the residual path F(x)
        - Preserve useful features via skip connections
        - Combine both for optimal representations
        
        This flexibility is why ResNets train so well!
        
        Args:
            x: Input tensor (N, 3, 224, 224)
            
        Returns:
            Output tensor (N, num_classes) with raw logits
        """
        # ========== STEM: Initial Processing ==========
        # Quick downsampling and feature extraction
        # 224×224×3 → 56×56×64
        x = self.stem(x)
        
        # ========== STAGE 1: Baseline Features ==========
        # No downsampling, establishes baseline feature representation
        # Spatial: 56×56, Channels: 64
        # Features: Low-level patterns, edges, colors, simple textures
        x = self.stage1(x)           # (N, 64, 56, 56)
        
        # ========== STAGE 2: Mid-level Features ==========
        # First downsampling (56→28), channel expansion (64→128)
        # Features: Corners, simple shapes, combined textures
        x = self.stage2(x)           # (N, 128, 28, 28)
        
        # ========== STAGE 3: Higher-level Features ==========
        # Second downsampling (28→14), channel expansion (128→256)
        # Features: Object parts, complex patterns, mid-level semantics
        x = self.stage3(x)           # (N, 256, 14, 14)
        
        # ========== STAGE 4: High-level Semantic Features ==========
        # Final downsampling (14→7), channel expansion (256→512)
        # Features: Complete objects, high-level concepts, semantic information
        x = self.stage4(x)           # (N, 512, 7, 7)
        
        # ========== GLOBAL POOLING: Spatial Aggregation ==========
        # Averages each 7×7 feature map to single value
        # Makes representation translation-invariant
        # (N, 512, 7, 7) → (N, 512, 1, 1)
        x = self.avgpool(x)
        
        # Flatten: (N, 512, 1, 1) → (N, 512)
        x = torch.flatten(x, 1)
        
        # ========== CLASSIFIER: Final Predictions ==========
        # Maps aggregated features to class scores
        # (N, 512) → (N, num_classes)
        x = self.fc(x)
        
        return x
    
if __name__ == "__main__":
    """
    Demonstration and analysis of ResNet-18 architecture.
    
    This example shows:
    1. Model instantiation and forward pass
    2. Parameter count and distribution
    3. Comparison with VGG
    4. Key architectural insights
    """
    
    print("=" * 70)
    print("ResNet-18: Deep Residual Learning Architecture Analysis")
    print("=" * 70)
    
    # Create model for ImageNet classification
    model = ResNet18(num_classes=1000)
    model.eval()
    
    # Create sample input
    x = torch.randn(1, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        y = model(x)
    
    print(f"\n[1] Input/Output Shapes")
    print(f"    Input:  {x.shape}")   # torch.Size([1, 3, 224, 224])
    print(f"    Output: {y.shape}")  # torch.Size([1, 1000])
    
    # Detailed parameter analysis
    print(f"\n[2] Parameter Distribution by Component")
    
    # Stem parameters
    stem_params = sum(p.numel() for p in model.stem.parameters())
    print(f"    Stem (Conv7×7 + BN):      {stem_params:>10,} params")
    
    # Stage parameters
    stage1_params = sum(p.numel() for p in model.stage1.parameters())
    stage2_params = sum(p.numel() for p in model.stage2.parameters())
    stage3_params = sum(p.numel() for p in model.stage3.parameters())
    stage4_params = sum(p.numel() for p in model.stage4.parameters())
    
    print(f"    Stage 1 (64 channels):    {stage1_params:>10,} params")
    print(f"    Stage 2 (128 channels):   {stage2_params:>10,} params")
    print(f"    Stage 3 (256 channels):   {stage3_params:>10,} params")
    print(f"    Stage 4 (512 channels):   {stage4_params:>10,} params")
    
    # Classifier parameters
    fc_params = sum(p.numel() for p in model.fc.parameters())
    print(f"    Classifier (FC):          {fc_params:>10,} params")
    
    # Total
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n    Total Parameters:         {total_params:>10,}")
    print(f"    Trainable Parameters:     {trainable_params:>10,}")
    
    # Calculate percentage distribution
    conv_params = total_params - fc_params
    print(f"\n    Convolutional Layers:     {conv_params:>10,} ({conv_params/total_params*100:.1f}%)")
    print(f"    Fully Connected:          {fc_params:>10,} ({fc_params/total_params*100:.1f}%)")
    
    # Architecture insights
    print(f"\n[3] Architectural Insights")
    print(f"    • Total depth: 18 weight layers (1 stem + 16 residual + 1 fc)")
    print(f"    • Skip connections: 8 residual blocks with identity/projection shortcuts")
    print(f"    • Spatial downsampling: 5× (1 stem + 4 stages)")
    print(f"    • Final spatial size: 7×7 (then global pooled to 1×1)")
    
    # Comparison with VGG
    print(f"\n[4] ResNet-18 vs VGG-16 Comparison")
    print(f"    {'Metric':<25} {'ResNet-18':<15} {'VGG-16':<15}")
    print(f"    {'-'*25} {'-'*15} {'-'*15}")
    print(f"    {'Parameters':<25} {'~11.7M':<15} {'~138M':<15}")
    print(f"    {'Depth':<25} {'18 layers':<15} {'16 layers':<15}")
    print(f"    {'Skip Connections':<25} {'Yes (8 blocks)':<15} {'No':<15}")
    print(f"    {'Batch Normalization':<25} {'Yes (built-in)':<15} {'Optional':<15}")
    print(f"    {'Pooling Strategy':<25} {'Global Avg':<15} {'Flatten':<15}")
    print(f"    {'FC Layer Size':<25} {'Small (~512K)':<15} {'Huge (~100M)':<15}")
    print(f"    {'Training Difficulty':<25} {'Easy':<15} {'Hard (deep)':<15}")
    
    print(f"\n[5] Why ResNet Works Better")
    print(f"    ✓ Skip connections enable training of very deep networks")
    print(f"    ✓ Gradients flow directly through shortcuts (no vanishing)")
    print(f"    ✓ Can learn identity mapping when deeper layers aren't needed")
    print(f"    ✓ Global avg pooling reduces parameters and overfitting")
    print(f"    ✓ 12× fewer parameters than VGG-16 but often more accurate")
    print(f"    ✓ Batch Normalization provides stable, fast training")
    
    print(f"\n[6] Key Innovation: Residual Learning")
    print(f"    Traditional: Learn H(x) directly")
    print(f"    ResNet:      Learn F(x) = H(x) - x, then compute H(x) = F(x) + x")
    print(f"    Benefit:     Easier to learn small adjustments (F(x) ≈ 0) than full transform")
    
    print("\n" + "=" * 70)
    print("ResNet revolutionized deep learning by enabling training of networks")
    print("with 100+ layers through simple but powerful skip connections!")
    print("=" * 70)
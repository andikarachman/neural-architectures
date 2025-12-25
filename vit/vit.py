"""
Vision Transformer (ViT): An Image is Worth 16x16 Words

Vision Transformer was introduced in "An Image is Worth 16x16 Words: Transformers 
for Image Recognition at Scale" by Dosovitskiy et al. (ICLR 2021, Google Research).

THE PARADIGM SHIFT: FROM CONVOLUTIONS TO ATTENTION
===================================================
Traditional Computer Vision (CNNs):
- Local receptive fields: Convolutions process small spatial neighborhoods
- Inductive biases: Translation equivariance, locality built into architecture
- Hierarchical features: Gradually build from edges → textures → objects
- Proven to work well with medium-sized datasets

Vision Transformer Approach:
- Global receptive fields: Self-attention sees entire image from layer 1
- Minimal inductive bias: Treats image as sequence of patches
- Learned position information: No built-in spatial structure
- Requires large-scale pre-training to work well

Key Insight: "Convolutions are not necessary for vision. With enough data,
             pure attention mechanisms can achieve excellent results."

THE CORE IDEA: IMAGES AS SEQUENCES
===================================
1. Split image into fixed-size patches (e.g., 16×16 pixels)
2. Flatten each patch into a 1D vector
3. Linearly project patches to embedding dimension
4. Add learnable position embeddings
5. Prepend a learnable [CLS] token
6. Process sequence through standard Transformer encoder
7. Use [CLS] token representation for classification

Example (224×224 image, 16×16 patches):
    Image: 224×224×3
      ↓
    Patches: 14×14 = 196 patches of size 16×16×3 = 768 dims each
      ↓
    Flatten + Project: 196 patches → 196 tokens of embedding_dim
      ↓
    Add [CLS] token: 197 tokens total
      ↓
    Add Position Embeddings: Learnable position encoding
      ↓
    Transformer Encoder: L layers of Multi-Head Self-Attention + MLP
      ↓
    Classification: MLP head on [CLS] token

ARCHITECTURE COMPONENTS:
========================

1. PATCH EMBEDDING:
   - Divide image into non-overlapping patches
   - Each patch is linearly projected to embedding dimension
   - Implementation: Single convolutional layer with kernel_size=patch_size, stride=patch_size
   - Alternative: Explicit reshape + linear projection

2. POSITION EMBEDDING:
   - 1D learnable position embeddings (not 2D!)
   - Added to patch embeddings to inject position information
   - Sine-cosine embeddings also work but learnable performed slightly better
   - Model must learn 2D spatial relationships from 1D positions

3. [CLS] TOKEN:
   - Special learnable embedding prepended to sequence
   - Serves as aggregate representation of entire image
   - Output of [CLS] token used for classification
   - Borrowed from BERT in NLP

4. TRANSFORMER ENCODER:
   - Standard architecture from "Attention is All You Need"
   - Each layer contains:
     a) Multi-Head Self-Attention (MSA)
     b) Layer Normalization (Pre-Norm: LN before MSA/MLP)
     c) Residual connections
     d) MLP with GELU activation
   - Pre-Norm (LN before attention) more stable than Post-Norm

5. MLP HEAD:
   - Simple classification head on [CLS] token
   - Usually one hidden layer with GELU, then output layer
   - During pre-training: May use different head than fine-tuning

MULTI-HEAD SELF-ATTENTION (MSA):
================================
For each token, compute attention with all other tokens:

1. Linear projections: Q = XW_q, K = XW_k, V = XW_v
2. Split into multiple heads (h heads of dimension d_k = d_model / h)
3. For each head: Attention(Q, K, V) = softmax(QK^T / √d_k) V
4. Concatenate heads and project: MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W_o

Key Properties:
- Computational complexity: O(n²·d) where n = sequence length
- Global receptive field: Every token attends to every other token
- Permutation equivariant (without positional encoding)
- Highly parallelizable

MLP BLOCK:
==========
Two-layer feed-forward network applied to each token independently:

    FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

- First layer: Expansion (d_model → mlp_ratio * d_model, typically 4×)
- GELU activation: Smooth non-linearity, better than ReLU for Transformers
- Second layer: Projection back (mlp_ratio * d_model → d_model)
- Provides non-linear transformation after attention

VIT MODEL VARIANTS:
===================
Paper introduces three main variants (parameters for patch_size=16):

Model      | Layers | Hidden Size | MLP Size | Heads | Params
-----------|--------|-------------|----------|-------|--------
ViT-Base   | 12     | 768         | 3072     | 12    | 86M
ViT-Large  | 24     | 1024        | 4096     | 16    | 307M
ViT-Huge   | 32     | 1280        | 5120     | 16    | 632M

Patch sizes: 16×16 (ViT-*/16) or 32×32 (ViT-*/32)
- Smaller patches = more tokens = better accuracy but higher compute
- 16×16 is standard, 32×32 for efficiency

TRAINING STRATEGY:
==================
The paper's key finding: ViT requires large-scale pre-training

1. PRE-TRAINING:
   - Large datasets: JFT-300M (proprietary), ImageNet-21k (14M images)
   - Simple objective: Classification on large label space
   - High resolution: Often 384×384 or higher
   - Data augmentation: RandAugment, Mixup, Cutmix

2. FINE-TUNING:
   - Transfer to downstream tasks (ImageNet-1k, CIFAR, etc.)
   - Higher resolution than pre-training (384×384 → 512×512)
   - Position embeddings interpolated for different resolutions
   - Few epochs needed (typically < 100)

3. KEY OBSERVATIONS:
   - Small/medium datasets (ImageNet-1k): CNNs outperform ViT
   - Large datasets (ImageNet-21k+): ViT matches or beats CNNs
   - Very large datasets (JFT-300M): ViT significantly outperforms CNNs
   - Reason: Transformers have less inductive bias, need more data to learn

INDUCTIVE BIASES COMPARISON:
=============================
CNNs:
- Locality: Convolutions only look at local neighborhoods
- Translation equivariance: Same weights applied everywhere
- Strong prior: Assumes nearby pixels are related

ViT:
- Minimal bias: Only patch-based structure, rest is learned
- Global context: Self-attention sees entire image from layer 1
- Weak prior: Must learn spatial relationships from data
- More flexible: Can learn different patterns for different tasks

ADVANTAGES OF VIT:
==================
1. Scalability: Transformers scale better with data/compute than CNNs
2. Transfer learning: Pre-trained ViT features transfer excellently
3. Simplicity: Standard Transformer architecture, no specialized components
4. Interpretability: Attention maps show what model focuses on
5. Unified architecture: Same model for images, text, multimodal tasks

LIMITATIONS:
============
1. Data hungry: Needs large pre-training datasets
2. Compute intensive: O(n²) attention complexity
3. Less efficient on small datasets: CNNs work better
4. Requires careful hyperparameter tuning
5. Longer training times than CNNs

PRACTICAL TIPS:
===============
1. Use pre-trained weights when possible (don't train from scratch on small data)
2. Fine-tune at higher resolution than pre-training
3. Use strong data augmentation (RandAugment, Mixup)
4. Pre-Norm (LN before attention) more stable than Post-Norm
5. Learning rate warmup essential for stable training
6. Adjust patch size based on compute budget (16 vs 32)

POSITION EMBEDDING INTERPOLATION:
==================================
When fine-tuning at different resolution:
- Pre-training: 224×224 → 14×14 patches → 196 position embeddings
- Fine-tuning: 384×384 → 24×24 patches → 576 position embeddings
- Solution: 2D interpolation of position embeddings
- Works because position embeddings learn smooth spatial patterns

SUBSEQUENT IMPROVEMENTS:
========================
After the original ViT paper, many improvements emerged:
- DeiT: Data-efficient training with distillation
- Swin Transformer: Shifted windows for hierarchical features
- CaiT: Going deeper with LayerScale
- BEiT: Masked image modeling pre-training
- MAE: Masked autoencoder pre-training (75% masking)

However, this implementation focuses on the original ViT architecture.

Reference:
    Alexey Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers 
    for Image Recognition at Scale." ICLR 2021.
    https://arxiv.org/abs/2010.11929
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    
    Converts a 2D image into a sequence of 1D patch embeddings, the fundamental
    transformation that enables treating images as sequences for Transformers.
    
    Process:
    1. Divide image into non-overlapping patches (e.g., 16×16)
    2. Flatten each patch into a vector
    3. Linearly project to embedding dimension
    
    Implementation:
    - Uses a single Conv2d with kernel_size=patch_size, stride=patch_size
    - This is equivalent to: split into patches → flatten → linear projection
    - But more efficient than explicit implementation
    
    Example (224×224 image, 16×16 patches, 768 embedding dim):
        Input: (B, 3, 224, 224)
        Conv2d: kernel=16, stride=16, out_channels=768
        Output: (B, 768, 14, 14)
        Flatten: (B, 768, 196)
        Transpose: (B, 196, 768)
    
    Args:
        img_size: Input image size (assumes square images)
        patch_size: Size of each patch (assumes square patches)
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Dimension of patch embeddings
    
    Shape:
        - Input: (batch_size, in_channels, img_size, img_size)
        - Output: (batch_size, num_patches, embed_dim)
        where num_patches = (img_size / patch_size)²
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # (224/16)² = 196
        
        # Convolutional projection: extracts patches and embeds them
        # Equivalent to splitting into patches, flattening, and linear projection
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - Batch of images
        Returns:
            (B, num_patches, embed_dim) - Sequence of patch embeddings
        """
        # Conv2d: (B, C, H, W) → (B, embed_dim, H/P, W/P)
        # where P = patch_size, H/P × W/P = num_patches
        x = self.projection(x)  # (B, 768, 14, 14) for standard ViT-Base
        
        # Flatten spatial dimensions: (B, embed_dim, H/P, W/P) → (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose to sequence format: (B, embed_dim, num_patches) → (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    The core operation of Transformers that allows each position to attend to
    all positions in the sequence, enabling global context understanding.
    
    Single Head Attention:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Multi-Head Attention:
        1. Project inputs to Q, K, V
        2. Split into h heads (each of dimension d_k = d_model / h)
        3. Apply attention in parallel for each head
        4. Concatenate outputs and project back
    
    Why Multiple Heads?
    - Different heads can learn different relationships
    - Head 1 might focus on local patterns
    - Head 2 might focus on long-range dependencies
    - Head 3 might focus on specific semantic relationships
    - Provides ensemble-like behavior within single layer
    
    Computational Complexity:
    - Self-attention: O(n² · d) where n = sequence length, d = embedding dim
    - For 196 patches: 196² ≈ 38k attention computations per layer
    - This is why ViT is more expensive than CNNs for high-resolution images
    
    Args:
        embed_dim: Total dimension of the model (e.g., 768 for ViT-Base)
        num_heads: Number of attention heads (e.g., 12 for ViT-Base)
        dropout: Dropout probability for attention weights
        bias: Whether to include bias in linear projections
    
    Shape:
        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)
    """
    def __init__(self, embed_dim: int, num_heads: int, 
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension per head
        self.scale = self.head_dim ** -0.5  # 1/√d_k for scaled dot-product
        
        # Linear projections for Q, K, V (combined for efficiency)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) where N = num_patches + 1 (includes [CLS] token)
        Returns:
            (B, N, D) - Attended features
        """
        B, N, D = x.shape
        
        # ========== 1. Linear Projections for Q, K, V ==========
        # Project input to Q, K, V all at once for efficiency
        # qkv: (B, N, D) → (B, N, 3*D)
        qkv = self.qkv(x)
        
        # Reshape to separate Q, K, V: (B, N, 3*D) → (B, N, 3, num_heads, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        
        # Permute to: (3, B, num_heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Extract Q, K, V: each is (B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # ========== 2. Scaled Dot-Product Attention ==========
        # Compute attention scores: Q @ K^T
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) 
        # → (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Softmax to get attention weights (sum to 1 across last dimension)
        attn = attn.softmax(dim=-1)
        
        # Apply dropout to attention weights (regularization)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values: attn @ V
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) 
        # → (B, num_heads, N, head_dim)
        x = attn @ v
        
        # ========== 3. Concatenate Heads and Project ==========
        # Transpose: (B, num_heads, N, head_dim) → (B, N, num_heads, head_dim)
        x = x.transpose(1, 2)
        
        # Reshape to concatenate heads: (B, N, num_heads, head_dim) → (B, N, D)
        x = x.reshape(B, N, D)
        
        # Final output projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network).
    
    Two-layer MLP with expansion, applied to each token independently (no
    interaction between tokens). Provides non-linear transformation after
    attention layer.
    
    Architecture:
        x → Linear(expand) → GELU → Dropout → Linear(project) → Dropout → out
    
    Typical expansion factor: 4×
        - ViT-Base: 768 → 3072 → 768
        - ViT-Large: 1024 → 4096 → 1024
    
    Why GELU instead of ReLU?
    - GELU: Smooth, differentiable everywhere
    - Formula: GELU(x) = x · Φ(x) where Φ is standard Gaussian CDF
    - Allows gradients for negative values (unlike ReLU)
    - Empirically works better for Transformers
    
    Args:
        in_features: Input dimension (embed_dim)
        hidden_features: Hidden layer dimension (typically 4 * in_features)
        out_features: Output dimension (typically same as in_features)
        dropout: Dropout probability
    
    Shape:
        - Input: (batch_size, seq_len, in_features)
        - Output: (batch_size, seq_len, out_features)
    """
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # Smooth activation function
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_features)
        Returns:
            (B, N, out_features)
        """
        x = self.fc1(x)        # Expand: (B, N, in_features) → (B, N, hidden_features)
        x = self.act(x)        # Non-linearity
        x = self.dropout(x)    # Regularization
        x = self.fc2(x)        # Project back: (B, N, hidden_features) → (B, N, out_features)
        x = self.dropout(x)    # Regularization
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block.
    
    The fundamental building block of Vision Transformer, consisting of:
    1. Multi-Head Self-Attention with residual connection
    2. MLP with residual connection
    3. Layer Normalization before each operation (Pre-Norm)
    
    Architecture (Pre-Norm design):
        x → LayerNorm → MultiHeadAttention → (+) → LayerNorm → MLP → (+) → out
        |___________________________|           |_______________________|
                Residual Connection                  Residual Connection
    
    Pre-Norm vs Post-Norm:
    - Pre-Norm: LN before attention/MLP (used in ViT)
        * More stable training
        * Easier to train very deep models
        * Better gradient flow
    
    - Post-Norm: LN after attention/MLP (original Transformer)
        * Slightly better performance when training converges
        * Harder to train deep models
        * Requires careful initialization
    
    Why Pre-Norm for ViT?
    - ViT has many layers (12-32), stability is crucial
    - Allows training without warm-up in some cases
    - Residual connections preserve original signal better
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Expansion ratio for MLP hidden dimension
        dropout: Dropout rate
        attn_dropout: Dropout rate for attention weights
    
    Shape:
        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        
        # Layer Normalization before attention (Pre-Norm)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Multi-Head Self-Attention
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout
        )
        
        # Layer Normalization before MLP (Pre-Norm)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP (Feed-Forward Network)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - Input sequence
        Returns:
            (B, N, D) - Output sequence
        """
        # ========== First Block: Attention with Residual ==========
        # Pre-Norm: Apply LayerNorm before attention
        # Residual: Add input to attention output
        x = x + self.attn(self.norm1(x))
        
        # ========== Second Block: MLP with Residual ==========
        # Pre-Norm: Apply LayerNorm before MLP
        # Residual: Add input to MLP output
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for Image Classification.
    
    Full Vision Transformer architecture that transforms images into sequences
    of patches and processes them with Transformer encoder layers.
    
    Architecture Pipeline:
    1. Patch Embedding: Image → Sequence of patch embeddings
    2. Add [CLS] Token: Prepend learnable classification token
    3. Add Position Embeddings: Inject positional information
    4. Transformer Encoder: Stack of L encoder blocks
    5. Extract [CLS] Token: Use first token as image representation
    6. Classification Head: MLP to predict class logits
    
    Complete Data Flow (ViT-Base example):
        Input Image: (B, 3, 224, 224)
          ↓ Patch Embedding
        Patches: (B, 196, 768)
          ↓ Add CLS Token
        Tokens: (B, 197, 768)  # 197 = 196 patches + 1 CLS
          ↓ Add Position Embeddings
        Tokens: (B, 197, 768)
          ↓ Transformer Encoder (12 layers)
        Encoded: (B, 197, 768)
          ↓ Extract CLS Token
        CLS: (B, 768)
          ↓ Classification Head
        Logits: (B, num_classes)
    
    Standard Configurations:
    
    ViT-Base/16:
        - Patch size: 16×16
        - Embedding: 768
        - Depth: 12 layers
        - Heads: 12
        - MLP ratio: 4
        - Params: ~86M
    
    ViT-Large/16:
        - Patch size: 16×16
        - Embedding: 1024
        - Depth: 24 layers
        - Heads: 16
        - MLP ratio: 4
        - Params: ~307M
    
    ViT-Huge/14:
        - Patch size: 14×14
        - Embedding: 1280
        - Depth: 32 layers
        - Heads: 16
        - MLP ratio: 4
        - Params: ~632M
    
    Args:
        img_size: Input image size (assumes square)
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of Transformer encoder blocks
        num_heads: Number of attention heads
        mlp_ratio: Expansion ratio for MLP
        dropout: Dropout rate
        attn_dropout: Dropout rate for attention weights
    
    Shape:
        - Input: (batch_size, in_channels, img_size, img_size)
        - Output: (batch_size, num_classes)
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # ========== 1. Patch Embedding ==========
        # Convert image to sequence of patch embeddings
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # ========== 2. [CLS] Token ==========
        # Learnable classification token prepended to sequence
        # Initialized randomly, learned during training
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # ========== 3. Position Embeddings ==========
        # Learnable 1D position embeddings
        # Shape: (1, num_patches + 1, embed_dim)
        # +1 accounts for [CLS] token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)
        
        # ========== 4. Transformer Encoder ==========
        # Stack of encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(depth)
        ])
        
        # ========== 5. Final Layer Norm ==========
        # Normalize before classification head
        self.norm = nn.LayerNorm(embed_dim)
        
        # ========== 6. Classification Head ==========
        # MLP head that maps [CLS] token to class logits
        # Can be single linear layer or small MLP
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights following the paper's approach.
        
        - Position embeddings: Normal distribution with std=0.02
        - CLS token: Normal distribution with std=0.02
        - Linear layers: Xavier uniform
        - Layer norms: weight=1, bias=0
        """
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize CLS token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize other parameters
        self.apply(self._init_module_weights)
        
    def _init_module_weights(self, m):
        """Initialize individual module weights."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Vision Transformer.
        
        Args:
            x: (B, C, H, W) - Batch of images
            
        Returns:
            (B, num_classes) - Class logits
        """
        B = x.shape[0]
        
        # ========== 1. Patch Embedding ==========
        # Convert image to sequence of patch embeddings
        # (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.patch_embed(x)  # (B, 196, 768) for ViT-Base
        
        # ========== 2. Prepend [CLS] Token ==========
        # Expand cls_token for batch: (1, 1, embed_dim) → (B, 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Concatenate: (B, 1, embed_dim) + (B, num_patches, embed_dim)
        #            → (B, num_patches + 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, 768)
        
        # ========== 3. Add Position Embeddings ==========
        # Add learnable position information to all tokens
        x = x + self.pos_embed  # Broadcasting: (B, 197, 768) + (1, 197, 768)
        x = self.pos_dropout(x)
        
        # ========== 4. Transformer Encoder ==========
        # Process through stack of encoder blocks
        # Each block: Self-Attention → MLP with residuals
        for block in self.blocks:
            x = block(x)  # (B, 197, 768) → (B, 197, 768)
        
        # ========== 5. Final Layer Norm ==========
        x = self.norm(x)  # (B, 197, 768)
        
        # ========== 6. Extract [CLS] Token ==========
        # Use first token as aggregate image representation
        cls_token_final = x[:, 0]  # (B, 768)
        
        # ========== 7. Classification Head ==========
        # Map [CLS] representation to class logits
        logits = self.head(cls_token_final)  # (B, num_classes)
        
        return logits


def vit_base_patch16_224(num_classes: int = 1000, **kwargs):
    """
    ViT-Base/16 with 224×224 input.
    
    Standard configuration from the paper.
    - 86M parameters
    - Good balance of accuracy and efficiency
    """
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs
    )


def vit_large_patch16_224(num_classes: int = 1000, **kwargs):
    """
    ViT-Large/16 with 224×224 input.
    
    Larger model for better accuracy with more compute.
    - 307M parameters
    """
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs
    )


def vit_huge_patch14_224(num_classes: int = 1000, **kwargs):
    """
    ViT-Huge/14 with 224×224 input.
    
    Largest standard ViT variant.
    - 632M parameters
    - Best accuracy but very compute intensive
    """
    return VisionTransformer(
        img_size=224,
        patch_size=14,
        num_classes=num_classes,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs
    )


if __name__ == "__main__":
    """
    Demonstration and analysis of Vision Transformer architecture.
    """
    
    print("=" * 80)
    print("Vision Transformer (ViT) - Architecture Analysis")
    print("=" * 80)
    
    # Create ViT-Base/16 model
    model = vit_base_patch16_224(num_classes=1000)
    model.eval()
    
    # Create sample input
    x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"\n[1] Input/Output Shapes")
    print(f"    Input:  {x.shape}")
    print(f"    Output: {output.shape}")
    
    # Architecture details
    print(f"\n[2] ViT-Base/16 Configuration")
    print(f"    Image size:       224×224")
    print(f"    Patch size:       16×16")
    print(f"    Number of patches: {model.num_patches} (14×14)")
    print(f"    Embedding dim:    {model.embed_dim}")
    print(f"    Number of layers: {len(model.blocks)}")
    print(f"    Number of heads:  {model.blocks[0].attn.num_heads}")
    print(f"    MLP ratio:        4.0")
    print(f"    Sequence length:  {model.num_patches + 1} (patches + CLS token)")
    
    # Parameter analysis
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n[3] Parameter Distribution")
    
    # Patch embedding
    patch_embed_params = sum(p.numel() for p in model.patch_embed.parameters())
    print(f"    Patch Embedding:        {patch_embed_params:>10,} params")
    
    # Position embeddings + CLS token
    pos_params = model.pos_embed.numel() + model.cls_token.numel()
    print(f"    Position Embed + CLS:   {pos_params:>10,} params")
    
    # Transformer blocks
    blocks_params = sum(p.numel() for p in model.blocks.parameters())
    print(f"    Transformer Blocks:     {blocks_params:>10,} params")
    
    # Classification head
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"    Classification Head:    {head_params:>10,} params")
    
    print(f"\n    Total Parameters:       {total_params:>10,}")
    print(f"    Trainable Parameters:   {trainable_params:>10,}")
    
    # Comparison with CNNs
    print(f"\n[4] ViT vs CNN Comparison")
    print(f"    {'Model':<20} {'Params':<12} {'ImageNet Top-1':<15}")
    print(f"    {'-'*20} {'-'*12} {'-'*15}")
    print(f"    {'ViT-Base/16':<20} {'86M':<12} {'~84.0% (JFT)':<15}")
    print(f"    {'ViT-Large/16':<20} {'307M':<12} {'~85.3% (JFT)':<15}")
    print(f"    {'ResNet-50':<20} {'26M':<12} {'~76.2%':<15}")
    print(f"    {'ResNet-152':<20} {'60M':<12} {'~78.3%':<15}")
    print(f"    {'EfficientNet-B7':<20} {'66M':<12} {'~84.3%':<15}")
    
    print(f"\n[5] Computational Complexity")
    num_patches = model.num_patches + 1  # +1 for CLS token
    print(f"    Self-Attention Complexity: O(n²·d) where n={num_patches}, d={model.embed_dim}")
    print(f"    Attention operations per layer: {num_patches}² = {num_patches**2:,}")
    print(f"    Total attention ops (12 layers): {12 * num_patches**2:,}")
    
    print(f"\n[6] Key Insights")
    print(f"    ✓ Pure Transformer architecture (no convolutions except patch embedding)")
    print(f"    ✓ Global receptive field from layer 1 (every token sees all tokens)")
    print(f"    ✓ Requires large-scale pre-training (ImageNet-21k or JFT-300M)")
    print(f"    ✓ Less inductive bias than CNNs (more flexible but needs more data)")
    print(f"    ✓ Excellent transfer learning capabilities")
    print(f"    ✓ Scales better with data/compute than CNNs")
    
    print(f"\n[7] Training Strategy")
    print(f"    Pre-training: Large datasets (JFT-300M, ImageNet-21k)")
    print(f"    Fine-tuning:  Downstream tasks with higher resolution")
    print(f"    Data Aug:     RandAugment, Mixup, Cutmix")
    print(f"    Optimizer:    AdamW with weight decay")
    print(f"    LR Schedule:  Cosine decay with linear warmup")
    
    print("\n" + "=" * 80)
    print("Vision Transformer revolutionized computer vision by showing that")
    print("Transformers can match or exceed CNN performance with enough data!")
    print("=" * 80)

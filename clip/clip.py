"""
CLIP: Connecting Text and Images

CLIP (Contrastive Language-Image Pre-training) was introduced in "Learning 
Transferable Visual Models From Natural Language Supervision" by Radford et al. 
(OpenAI, 2021).

THE PARADIGM SHIFT: LEARNING FROM NATURAL LANGUAGE
===================================================
Traditional Computer Vision:
- Supervised learning on fixed label sets (ImageNet's 1000 classes)
- Requires expensive manual annotation
- Poor zero-shot transfer to new tasks
- Limited to predefined categories

CLIP Approach:
- Learn from (image, text) pairs scraped from the internet
- 400 million (image, caption) pairs from web
- No manual labeling required
- Learns visual concepts from natural language descriptions
- Zero-shot transfer to many downstream tasks

Key Insight: "Natural language provides a flexible and rich supervision signal
             that enables learning visual representations that generalize broadly."

THE CORE IDEA: CONTRASTIVE LEARNING
====================================
Traditional approach: Predict exact caption for each image (generative)
- Complex: Must generate coherent text
- Slow: Sequential text generation
- Hard to optimize

CLIP approach: Match images and captions via contrastive learning
- Simple: Learn which image-text pairs match
- Fast: Single forward pass, no generation
- Efficient: Symmetric loss for both modalities

Training Process:
1. Start with batch of N (image, text) pairs
2. Encode all images → image embeddings (N × d)
3. Encode all texts → text embeddings (N × d)
4. Compute similarity matrix: images @ texts.T → (N × N)
5. Train to maximize diagonal (correct pairs), minimize off-diagonal (wrong pairs)
6. Use symmetric cross-entropy loss

Result: 
- Diagonal = high similarity (cat image ↔ "a photo of a cat")
- Off-diagonal = low similarity (cat image ↔ "a photo of a dog")

CONTRASTIVE LOSS (InfoNCE):
===========================
For each image-text pair (i, t):

    L_i→t = -log( exp(sim(i,t)/τ) / Σ_j exp(sim(i,j)/τ) )
    L_t→i = -log( exp(sim(t,i)/τ) / Σ_j exp(sim(j,i)/τ) )
    
    L_total = (L_i→t + L_t→i) / 2

Where:
- sim(i,t) = cosine_similarity(image_embed, text_embed) = (i·t) / (||i|| ||t||)
- τ = temperature parameter (learnable, typically ~0.07)
- Symmetric: Train both image→text and text→image directions

Why it works:
- Image learns to be similar to its caption
- Image learns to be dissimilar to other captions in batch
- Larger batch = more negative examples = better learning
- Temperature τ controls sharpness of distribution

ARCHITECTURE COMPONENTS:
========================

1. IMAGE ENCODER:
   Two options in the paper:
   
   a) Vision Transformer (ViT):
      - Standard ViT architecture (Base, Large, or Huge)
      - Input: 224×224 or higher resolution images
      - Output: [CLS] token embedding
      - Performs better than ResNet variants
   
   b) ResNet (Modified):
      - Standard ResNet with attention pooling instead of average pooling
      - Replaces global average pooling with attention-based pooling
      - Output: Attention-weighted features
      - More compute efficient than ViT

2. TEXT ENCODER:
   Transformer architecture (similar to GPT-2):
   - Input: Tokenized text (max 77 tokens)
   - 12 layers of Transformer decoder blocks
   - Causal masking (can only attend to previous tokens)
   - Output: Embedding from [EOS] token position
   - 63M parameters for text encoder
   
   Vocabulary: 49,152 tokens (BPE encoding)

3. PROJECTION HEADS:
   - Image projection: Maps image embedding to joint space
   - Text projection: Maps text embedding to joint space
   - Both project to same dimensionality (typically 512 or 1024)
   - Enables direct comparison via cosine similarity

4. TEMPERATURE PARAMETER:
   - Learnable scalar τ (initialized to 0.07)
   - Controls distribution sharpness in contrastive loss
   - Lower τ → sharper distributions, harder negatives
   - Higher τ → softer distributions, easier learning

ZERO-SHOT CLASSIFICATION:
==========================
How CLIP performs classification without training on the task:

1. Define text prompts for each class:
   Classes: ["cat", "dog", "bird"]
   Prompts: ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

2. Encode all prompts → text embeddings (C × d)

3. For test image:
   - Encode image → image embedding (1 × d)
   - Compute similarity with all text embeddings
   - Predict class with highest similarity
   
4. Key: Uses natural language as the "classifier weights"
   - No task-specific training needed
   - Can add new classes just by defining prompts
   - Prompt engineering matters!

PROMPT ENGINEERING:
===================
Performance varies significantly with prompt design:

Simple: "cat" → Lower accuracy
Better: "a photo of a cat" → Good accuracy
Best: Ensemble of prompts → Best accuracy
  - "a photo of a cat"
  - "a picture of a cat"
  - "an image of a cat"
  - etc.

The paper uses 80 prompt templates and averages their embeddings.

TRAINING DETAILS:
=================
Dataset: 400M (image, text) pairs from internet
- Scraped from publicly available web sources
- No manual labeling
- Diverse: Many domains, languages, concepts

Optimization:
- Batch size: 32,768 (very large!)
- Optimizer: AdamW
- Learning rate: 5e-4 with cosine decay
- Training: 32 epochs on 400M examples
- Compute: 592 V100 GPUs for 12 days

Data augmentation:
- Images: Random crop, resize
- Text: No augmentation (use as-is)

Mixed precision training:
- Crucial for training at this scale
- FP16 for speed, FP32 for stability

MODEL VARIANTS:
===============
CLIP was trained with multiple encoder combinations:

Image Encoders:
- ResNet-50 (RN50)
- ResNet-101 (RN101)
- ViT-B/32 (ViT-Base, patch_size=32)
- ViT-B/16 (ViT-Base, patch_size=16)
- ViT-L/14 (ViT-Large, patch_size=14)

Text Encoder:
- 12-layer Transformer (63M params)
- Same for all variants

Best performing: ViT-L/14
- Highest accuracy on most benchmarks
- Most compute intensive

PERFORMANCE HIGHLIGHTS:
=======================
Zero-shot ImageNet Classification:
- ResNet-50: ~60% top-1 accuracy
- ViT-B/32: ~63% top-1 accuracy
- ViT-L/14: ~76% top-1 accuracy

Compare to supervised baselines:
- ResNet-50 (supervised): ~76% (with task-specific training!)
- CLIP ViT-L/14 (zero-shot): ~76% (no task-specific training!)

Other tasks (zero-shot):
- Object detection: Competitive with supervised methods
- Action recognition: Strong performance
- OCR: Excellent text reading
- Geo-localization: Reasonable performance

ADVANTAGES OF CLIP:
===================
1. Zero-shot transfer: Works on new tasks without training
2. Flexible: Use natural language to define tasks
3. Robust: Handles distribution shift better than supervised models
4. Multi-modal: Understands both images and text
5. Scalable: Performance improves with more data/compute
6. Efficient: No need for expensive labeled datasets

LIMITATIONS:
============
1. Data hungry: Requires 400M image-text pairs
2. Compute intensive: Large batches needed (32k)
3. Abstract tasks: Struggles with counting, spatial reasoning
4. Fine-grained: Less accurate than specialized supervised models
5. Bias: Inherits biases from web data
6. Out-of-distribution: Can fail on very different distributions

APPLICATIONS:
=============
1. Zero-shot classification: No training data needed
2. Image search: Text queries to find images
3. Content moderation: Detect inappropriate content
4. Visual question answering: Combine with LLM
5. Image generation guidance: DALL-E, Stable Diffusion use CLIP
6. Multi-modal embeddings: Joint representation space

SUBSEQUENT WORK:
================
CLIP inspired many follow-up papers:
- OpenCLIP: Open-source replication
- ALIGN: Google's approach with noisy data
- Florence: Microsoft's large-scale vision foundation model
- BLIP: Bootstrapping language-image pre-training
- EVA-CLIP: Improved training and scaling

However, this implementation focuses on the original CLIP architecture.

KEY TAKEAWAYS:
==============
1. Natural language supervision is powerful and scalable
2. Contrastive learning is more efficient than generative modeling
3. Large-scale pre-training enables zero-shot transfer
4. Joint embedding spaces unify vision and language
5. Simple methods can work exceptionally well at scale

Reference:
    Alec Radford et al. "Learning Transferable Visual Models From Natural 
    Language Supervision." arXiv:2103.00020, 2021.
    https://arxiv.org/abs/2103.00020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention for text encoder.
    
    Standard Transformer attention mechanism with causal masking for text.
    Identical to attention in GPT models.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - Input sequence
            attn_mask: (N, N) - Causal mask for text (optional)
        Returns:
            (B, N, D) - Attended output
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)                                         # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim) # (B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                          # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # Each: (B, num_heads, N, head_dim)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale             # (B, num_heads, N, N)
        
        # Apply causal mask if provided (for text)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))# (B, num_heads, N, N)
        
        attn = attn.softmax(dim=-1)                               # (B, num_heads, N, N)
        attn = self.attn_dropout(attn)                            # (B, num_heads, N, N)
        
        # Apply attention to values
        x = attn @ v                                              # (B, num_heads, N, head_dim)
        x = x.transpose(1, 2)                                     # (B, N, num_heads, head_dim)
        x = x.reshape(B, N, D)                                    # (B, N, D)
        x = self.proj(x)                                          # (B, N, D)
        x = self.proj_dropout(x)                                  # (B, N, D)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and MLP.
    
    Standard Transformer architecture used in both image and text encoders.
    Pre-LayerNorm design (LN before attention/MLP).
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout probability
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            attn_mask: Optional attention mask for causal masking
        Returns:
            (B, N, D)
        """
        # Attention with residual
        x = self.ln1(x)
        x = x + self.attn(x, attn_mask)
        
        # MLP with residual
        x = self.ln2(x)
        x = x + self.mlp(x)
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for CLIP image encoder.
    
    Standard ViT architecture as described in the CLIP paper.
    Takes images and outputs embeddings projected to joint space.
    
    Architecture matches the ViT described in "An Image is Worth 16x16 Words"
    with modifications for CLIP:
    - Final layer norm before projection
    - Projects to joint embedding space
    
    Args:
        img_size: Input image size
        patch_size: Size of image patches
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        output_dim: Output projection dimension (joint space)
        dropout: Dropout probability
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        output_dim: int = 512,
        dropout: float = 0.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Projection to joint embedding space
        self.proj = nn.Linear(embed_dim, output_dim, bias=False)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - Batch of images
        Returns:
            (B, output_dim) - Image embeddings in joint space
        """
        B = x.shape[0]
        
        # Patch embedding: (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.patch_embed(x)                     # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)            # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)       # (B, num_patches + 1, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed                      # (B, num_patches + 1, embed_dim)
        x = self.pos_dropout(x)                     # (B, num_patches + 1, embed_dim)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)                            # (B, num_patches + 1, embed_dim)
        
        # Extract CLS token and normalize
        x = self.ln_final(x[:, 0])                  # (B, embed_dim)
        
        # Project to joint space
        x = self.proj(x)                            # (B, output_dim)   
        
        return x


class TextTransformer(nn.Module):
    """
    Text Transformer encoder for CLIP.
    
    Causal Transformer similar to GPT-2 architecture. Processes tokenized
    text and outputs embeddings projected to joint space.
    
    Key differences from standard GPT:
    - Uses [EOS] token embedding instead of last token
    - Projects to joint embedding space with image encoder
    - Trained with contrastive loss, not language modeling
    
    Architecture:
    - Token embedding + position embedding
    - 12 Transformer layers with causal masking
    - Final layer norm
    - Projection to joint space
    
    Args:
        vocab_size: Size of vocabulary (typically 49,152 for CLIP)
        context_length: Maximum sequence length (77 for CLIP)
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        output_dim: Output projection dimension (joint space)
        dropout: Dropout probability
    """
    def __init__(
        self,
        vocab_size: int = 49152,
        context_length: int = 77,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        output_dim: int = 512,
        dropout: float = 0.0
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, context_length, embed_dim))
        
        # Transformer blocks with causal masking
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Projection to joint embedding space
        self.proj = nn.Linear(embed_dim, output_dim, bias=False)
        
        # Create causal mask (lower triangular)
        # Tokens can only attend to previous tokens
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_length, context_length)).view(
                1, 1, context_length, context_length
            )
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.01)
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text: (B, L) - Tokenized text sequences (L = context_length)
        Returns:
            (B, output_dim) - Text embeddings in joint space
        """
        B, L = text.shape
        
        # Token embeddings + position embeddings
        x = self.token_embedding(text)  # (B, L, embed_dim)
        x = x + self.pos_embedding[:, :L, :]
        
        # Transformer blocks with causal masking
        for block in self.blocks:
            x = block(x, attn_mask=self.causal_mask[:, :, :L, :L])
        
        # Layer norm
        x = self.ln_final(x)
        
        # Extract features at [EOS] token position
        # CLIP uses the embedding at the [EOS] token position
        # We'll use the last token for simplicity (argmax finds [EOS])
        # In practice, you'd pass eot_token position
        x = x[torch.arange(B), text.argmax(dim=-1)]  # (B, embed_dim)
        
        # Project to joint space
        x = self.proj(x)                            # (B, output_dim)
        
        return x


class CLIP(nn.Module):
    """
    CLIP: Contrastive Language-Image Pre-training.
    
    Learns to match images and their textual descriptions using contrastive
    learning. Creates a joint embedding space where semantically similar
    image-text pairs are close together.
    
    Architecture:
    ┌─────────────┐         ┌─────────────┐
    │   Image     │         │    Text     │
    │   Encoder   │         │   Encoder   │
    │   (ViT)     │         │ (Transformer)│
    └──────┬──────┘         └──────┬──────┘
           │                       │
           │ Project               │ Project
           ▼                       ▼
    ┌─────────────┐         ┌─────────────┐
    │   Image     │         │    Text     │
    │ Embeddings  │◄───────►│ Embeddings  │
    │  (B, 512)   │  Cosine │  (B, 512)   │
    │             │ Similarity│             │
    └─────────────┘         └─────────────┘
           │                       │
           └───────────┬───────────┘
                       ▼
              Contrastive Loss
         (Maximize diagonal, minimize off-diagonal)
    
    Training:
    1. Encode batch of N images → image embeddings
    2. Encode batch of N texts → text embeddings
    3. Compute similarity matrix: images @ texts.T (N × N)
    4. Maximize similarity for correct pairs (diagonal)
    5. Minimize similarity for incorrect pairs (off-diagonal)
    
    Zero-shot Inference:
    1. Encode image
    2. Encode multiple class descriptions
    3. Compute similarities
    4. Predict class with highest similarity
    
    Args:
        embed_dim: Dimension of joint embedding space
        image_resolution: Input image size
        vision_layers: Number of ViT layers
        vision_width: ViT embedding dimension
        vision_patch_size: ViT patch size
        context_length: Max text sequence length
        vocab_size: Text vocabulary size
        text_layers: Number of text transformer layers
        text_width: Text embedding dimension
        text_heads: Number of text attention heads
    """
    def __init__(
        self,
        embed_dim: int = 512,
        # Image encoder params
        image_resolution: int = 224,
        vision_layers: int = 12,
        vision_width: int = 768,
        vision_patch_size: int = 16,
        # Text encoder params
        context_length: int = 77,
        vocab_size: int = 49152,
        text_layers: int = 12,
        text_width: int = 512,
        text_heads: int = 8,
    ):
        super().__init__()
        
        self.context_length = context_length
        self.vocab_size = vocab_size
        
        # ========== IMAGE ENCODER ==========
        # Vision Transformer for encoding images
        self.visual = VisionTransformer(
            img_size=image_resolution,
            patch_size=vision_patch_size,
            embed_dim=vision_width,
            depth=vision_layers,
            num_heads=vision_width // 64,  # Standard: 64 dims per head
            output_dim=embed_dim
        )
        
        # ========== TEXT ENCODER ==========
        # Transformer for encoding text
        self.text = TextTransformer(
            vocab_size=vocab_size,
            context_length=context_length,
            embed_dim=text_width,
            depth=text_layers,
            num_heads=text_heads,
            output_dim=embed_dim
        )
        
        # ========== TEMPERATURE PARAMETER ==========
        # Learnable temperature for contrastive loss
        # Initialized to log(1/0.07) ≈ 2.66 (so exp(logit_scale) ≈ 14.3)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialize weights following CLIP paper specifications.
        
        Based on the official OpenAI CLIP implementation:
        https://github.com/openai/CLIP/blob/main/clip/model.py#L299-L326
        
        Key initializations:
        1. Token embedding: std=0.02 (done in TextTransformer)
        2. Position embeddings: std=0.01 (done in TextTransformer and VisionTransformer)
        3. Transformer attention/MLP: Layer-dependent initialization
        4. Text projection: std = transformer_width ** -0.5
        
        The layer-dependent initialization accounts for:
        - Depth of the network (number of layers)
        - Width of the network (embedding dimension)
        - Residual connections (scaling factor)
        
        Formula:
        - proj_std = (transformer_width ** -0.5) * ((2 * num_layers) ** -0.5)
        - attn_std = transformer_width ** -0.5
        - fc_std = (2 * transformer_width) ** -0.5
        
        This ensures:
        - Gradients remain stable through many layers
        - Activations don't explode or vanish
        - Each residual branch contributes appropriately
        """
        # ========== TEXT ENCODER INITIALIZATION ==========
        # Initialize transformer blocks with layer-dependent std
        text_width = self.text.token_embedding.embedding_dim
        text_layers = len(self.text.blocks)
        
        # Scaling factors for different components
        proj_std = (text_width ** -0.5) * ((2 * text_layers) ** -0.5)
        attn_std = text_width ** -0.5
        fc_std = (2 * text_width) ** -0.5
        
        for block in self.text.blocks:
            # Initialize attention weights
            nn.init.normal_(block.attn.qkv.weight, std=attn_std)
            nn.init.normal_(block.attn.proj.weight, std=proj_std)
            
            # Initialize MLP weights
            nn.init.normal_(block.mlp[0].weight, std=fc_std)    # First linear layer
            nn.init.normal_(block.mlp[3].weight, std=proj_std)  # Second linear layer
        
        # Initialize text projection to joint space
        # std = text_width ** -0.5 ensures proper scale in joint embedding space
        nn.init.normal_(self.text.proj.weight, std=text_width ** -0.5)
        
        # Note: Vision transformer weights use default PyTorch initialization
        # (Kaiming uniform for Conv2d, normal with std=0.02 for embeddings)
        # This follows the original CLIP implementation
        
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode images to joint embedding space.
        
        Args:
            image: (B, 3, H, W) - Batch of images
        Returns:
            (B, embed_dim) - L2-normalized image embeddings
        """
        features = self.visual(image)
        # L2 normalize embeddings (crucial for contrastive learning)
        features = F.normalize(features, dim=-1)
        return features
        
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """
        Encode text to joint embedding space.
        
        Args:
            text: (B, context_length) - Tokenized text
        Returns:
            (B, embed_dim) - L2-normalized text embeddings
        """
        features = self.text(text)
        # L2 normalize embeddings (crucial for contrastive learning)
        features = F.normalize(features, dim=-1)
        return features
        
    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Computes image and text features, then their similarity matrix.
        The contrastive loss maximizes diagonal (correct pairs) and
        minimizes off-diagonal (incorrect pairs).
        
        Args:
            image: (B, 3, H, W) - Batch of images
            text: (B, context_length) - Batch of tokenized texts
            
        Returns:
            image_features: (B, embed_dim) - Normalized image embeddings
            text_features: (B, embed_dim) - Normalized text embeddings
            logit_scale: Scalar temperature parameter (exp of learned param)
        """
        # Encode both modalities
        image_features = self.encode_image(image)  # (B, embed_dim)
        text_features = self.encode_text(text)     # (B, embed_dim)
        
        return image_features, text_features, self.logit_scale.exp()
    
    def get_similarity(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix between images and texts.
        
        Used for zero-shot inference: compute similarities between
        an image and multiple text descriptions.
        
        Args:
            image: (B, 3, H, W) or (B, embed_dim) - Images or image features
            text: (N, context_length) or (N, embed_dim) - Texts or text features
            
        Returns:
            (B, N) - Similarity matrix (scaled by temperature)
        """
        # Encode if needed
        if image.dim() == 4:
            image_features = self.encode_image(image)
        else:
            image_features = F.normalize(image, dim=-1)
            
        if text.dim() == 2 and text.dtype == torch.long:
            text_features = self.encode_text(text)
        else:
            text_features = F.normalize(text, dim=-1)
        
        # Cosine similarity (dot product of normalized vectors)
        # Shape: (B, embed_dim) @ (embed_dim, N) = (B, N)
        similarity = image_features @ text_features.T
        
        # Scale by temperature
        similarity = similarity * self.logit_scale.exp()
        
        return similarity


def clip_loss(image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """
    CLIP contrastive loss (symmetric cross-entropy).
    
    Computes the symmetric contrastive loss used in CLIP training.
    The loss encourages:
    - High similarity between matched image-text pairs (diagonal)
    - Low similarity between unmatched pairs (off-diagonal)
    
    Mathematical formulation:
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        labels = [0, 1, 2, ..., B-1] (identity matrix positions)
        
        loss_i = CrossEntropy(logits_per_image, labels)
        loss_t = CrossEntropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
    
    Why symmetric?
    - Trains both directions: image→text and text→image
    - More stable training
    - Better performance
    
    Args:
        image_features: (B, embed_dim) - L2-normalized image embeddings
        text_features: (B, embed_dim) - L2-normalized text embeddings
        logit_scale: Scalar - Temperature parameter (exp of learned value)
        
    Returns:
        Scalar loss value
    """
    # Compute similarity matrices
    # (B, embed_dim) @ (embed_dim, B) = (B, B)
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logit_scale * text_features @ image_features.T
    
    # Labels are just the diagonal indices (correct pairs)
    # For batch of 4: labels = [0, 1, 2, 3]
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    
    # Symmetric cross-entropy loss
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2
    
    return loss


def clip_vit_base_patch16():
    """
    CLIP with ViT-B/16 image encoder.
    
    Configuration:
    - Image: ViT-Base with 16×16 patches
    - Text: 12-layer Transformer
    - Joint embedding: 512 dimensions
    - Parameters: ~150M total
    """
    return CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=16,
        context_length=77,
        vocab_size=49152,
        text_layers=12,
        text_width=512,
        text_heads=8
    )


def clip_vit_large_patch14():
    """
    CLIP with ViT-L/14 image encoder.
    
    Best performing CLIP variant in the paper.
    
    Configuration:
    - Image: ViT-Large with 14×14 patches
    - Text: 12-layer Transformer
    - Joint embedding: 768 dimensions
    - Parameters: ~428M total
    """
    return CLIP(
        embed_dim=768,
        image_resolution=224,
        vision_layers=24,
        vision_width=1024,
        vision_patch_size=14,
        context_length=77,
        vocab_size=49152,
        text_layers=12,
        text_width=512,
        text_heads=8
    )


if __name__ == "__main__":
    """
    Demonstration of CLIP architecture and zero-shot inference.
    """
    
    print("=" * 80)
    print("CLIP: Contrastive Language-Image Pre-training")
    print("=" * 80)
    
    # Create CLIP model
    model = clip_vit_base_patch16()
    model.eval()
    
    # Sample data
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    texts = torch.randint(0, 49152, (batch_size, 77))  # Random token IDs
    
    print(f"\n[1] Model Architecture")
    print(f"    Image Encoder: Vision Transformer (ViT-B/16)")
    print(f"    Text Encoder:  12-layer Transformer")
    print(f"    Joint Space:   512 dimensions")
    
    # Forward pass
    with torch.no_grad():
        image_features, text_features, logit_scale = model(images, texts)
    
    print(f"\n[2] Encoding Examples")
    print(f"    Input images:   {images.shape}")
    print(f"    Input texts:    {texts.shape}")
    print(f"    Image features: {image_features.shape}")
    print(f"    Text features:  {text_features.shape}")
    print(f"    Temperature τ:  {1/logit_scale.item():.4f}")
    
    # Compute similarity matrix
    similarity = image_features @ text_features.T * logit_scale
    
    print(f"\n[3] Similarity Matrix")
    print(f"    Shape: {similarity.shape} (batch_size × batch_size)")
    print(f"    Diagonal (correct pairs):     {similarity.diag().mean():.3f} ± {similarity.diag().std():.3f}")
    
    # Off-diagonal mask
    mask = torch.ones_like(similarity).fill_diagonal_(0).bool()
    off_diag = similarity[mask]
    print(f"    Off-diagonal (incorrect pairs): {off_diag.mean():.3f} ± {off_diag.std():.3f}")
    
    # Compute loss
    loss = clip_loss(image_features, text_features, logit_scale)
    print(f"\n[4] Contrastive Loss: {loss.item():.4f}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    vision_params = sum(p.numel() for p in model.visual.parameters())
    text_params = sum(p.numel() for p in model.text.parameters())
    
    print(f"\n[5] Parameter Distribution")
    print(f"    Image Encoder:  {vision_params:>12,} params ({vision_params/total_params*100:.1f}%)")
    print(f"    Text Encoder:   {text_params:>12,} params ({text_params/total_params*100:.1f}%)")
    print(f"    Total:          {total_params:>12,} params")
    
    # Zero-shot classification example
    print(f"\n[6] Zero-Shot Classification Example")
    print(f"    Given an image and class descriptions:")
    print(f"    1. Encode image → image_embedding")
    print(f"    2. Encode texts → text_embeddings for each class")
    print(f"    3. Compute similarities")
    print(f"    4. Predict class with highest similarity")
    
    # Simulate zero-shot inference
    num_classes = 5
    image = torch.randn(1, 3, 224, 224)
    class_texts = torch.randint(0, 49152, (num_classes, 77))
    
    with torch.no_grad():
        similarity = model.get_similarity(image, class_texts)
        probs = similarity.softmax(dim=-1)
        predicted_class = probs.argmax(dim=-1)
    
    print(f"\n    Image shape:        {image.shape}")
    print(f"    Class texts shape:  {class_texts.shape}")
    print(f"    Similarities:       {similarity[0].tolist()}")
    print(f"    Probabilities:      {probs[0].tolist()}")
    print(f"    Predicted class:    {predicted_class.item()}")
    
    print(f"\n[7] Key Advantages")
    print(f"    ✓ Zero-shot transfer: No task-specific training needed")
    print(f"    ✓ Flexible: Use natural language to define tasks")
    print(f"    ✓ Multi-modal: Joint understanding of vision and language")
    print(f"    ✓ Robust: Better handling of distribution shift")
    print(f"    ✓ Scalable: Performance improves with more data")
    
    print(f"\n[8] Training Details")
    print(f"    Dataset:     400M (image, text) pairs from internet")
    print(f"    Batch size:  32,768 (very large!)")
    print(f"    Epochs:      32")
    print(f"    Compute:     592 V100 GPUs × 12 days")
    print(f"    Loss:        Symmetric contrastive loss (InfoNCE)")
    
    print(f"\n[9] Zero-Shot ImageNet Performance")
    print(f"    ViT-B/32:  ~63% top-1 accuracy")
    print(f"    ViT-B/16:  ~68% top-1 accuracy")
    print(f"    ViT-L/14:  ~76% top-1 accuracy (best)")
    print(f"    Compare: ResNet-50 supervised = ~76% (needs training!)")
    
    print("\n" + "=" * 80)
    print("CLIP revolutionized vision by learning from natural language supervision,")
    print("enabling zero-shot transfer and multi-modal understanding!")
    print("=" * 80)

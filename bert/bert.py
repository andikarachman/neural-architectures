"""
BERT: Bidirectional Encoder Representations from Transformers

BERT was introduced in "BERT: Pre-training of Deep Bidirectional Transformers 
for Language Understanding" by Devlin et al. (Google AI Language, 2018).

THE PARADIGM SHIFT: BIDIRECTIONAL PRE-TRAINING
===============================================
Traditional Language Models (GPT, ELMo):
- Unidirectional: Left-to-right or right-to-left
- Cannot fully understand context from both directions
- Limited representation power

BERT's Innovation:
- Bidirectional: Sees both left AND right context simultaneously
- Pre-trained on massive unlabeled text (Wikipedia + BookCorpus)
- Fine-tuned on downstream tasks (classification, QA, NER, etc.)
- Transformer Encoder only (no decoder needed)

Key Insight: "Deep bidirectional representations are more powerful than 
             left-to-right or shallow bidirectional combinations."

THE CORE ARCHITECTURE: TRANSFORMER ENCODER
==========================================
BERT uses only the encoder part of the Transformer:

                    [Input Tokens]
                          ↓
              [Token + Segment + Position Embeddings]
                          ↓
                 ┌─────────────────┐
                 │  Layer Norm     │
                 │  Multi-Head     │
                 │  Self-Attention │  ← Bidirectional!
                 │  + Residual     │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │  Layer Norm     │
                 │  Feed-Forward   │
                 │  + Residual     │
                 └────────┬────────┘
                          ↓
                    × N Layers
                          ↓
                  [Output Embeddings]

Key difference from GPT:
- GPT: Causal attention (can only see previous tokens)
- BERT: Full attention (can see all tokens in sequence)

BERT VARIANTS:
==============
BERT-Base:
- 12 layers (Transformer blocks)
- 768 hidden dimensions
- 12 attention heads
- 110M parameters
- Max sequence length: 512 tokens

BERT-Large:
- 24 layers
- 1024 hidden dimensions  
- 16 attention heads
- 340M parameters
- Max sequence length: 512 tokens

PRE-TRAINING TASKS:
===================
BERT is pre-trained using TWO unsupervised tasks simultaneously:

1. MASKED LANGUAGE MODEL (MLM):
   --------------------------------
   Traditional LM: Predict next word (unidirectional)
   BERT MLM: Mask some tokens, predict them (bidirectional)
   
   Example:
   Input:  "My dog is [MASK]. He likes [MASK]."
   Target: "My dog is hairy. He likes playing."
   
   Process:
   - Randomly mask 15% of tokens
   - Of those masked tokens:
     * 80% replaced with [MASK]
     * 10% replaced with random word
     * 10% kept unchanged
   
   Why this works:
   - Forces model to understand context from BOTH directions
   - Random replacement prevents overfitting to [MASK] token
   - Unchanged tokens teach model about all positions
   
   Loss: Cross-entropy on masked token predictions

2. NEXT SENTENCE PREDICTION (NSP):
   ---------------------------------
   Task: Predict if sentence B follows sentence A
   
   Example:
   Input A: "The man went to the store."
   Input B: "He bought a gallon of milk."
   Label: IsNext (True)
   
   Input A: "The man went to the store."
   Input B: "Penguins are flightless birds."
   Label: NotNext (False)
   
   Format:
   [CLS] Sentence A [SEP] Sentence B [SEP]
   
   Why this helps:
   - Learns relationships between sentences
   - Crucial for QA and NLI tasks
   - [CLS] token learns sentence-pair representation
   
   Loss: Binary cross-entropy on [CLS] token output

Combined Loss:
   L_total = L_MLM + L_NSP

INPUT REPRESENTATION:
=====================
BERT uses three types of embeddings summed together:

1. TOKEN EMBEDDINGS:
   - WordPiece tokenization (30,000 vocab)
   - Handles rare words via subword units
   - Example: "playing" → "play" + "##ing"

2. SEGMENT EMBEDDINGS:
   - Indicates which sentence token belongs to
   - Segment A (first sentence): embedding E_A
   - Segment B (second sentence): embedding E_B
   - Helps distinguish sentence pairs

3. POSITION EMBEDDINGS:
   - Learned (not sinusoidal like original Transformer)
   - Position 0, 1, 2, ..., 511
   - Each position has unique learned embedding

Final Input:
   Input = Token_Emb + Segment_Emb + Position_Emb

Special Tokens:
- [CLS]: Classification token (always first)
  * Used for sequence-level tasks
  * Final hidden state represents entire input
  
- [SEP]: Separator token
  * Separates sentences in sentence pairs
  * Marks end of sequence
  
- [MASK]: Masked token (for MLM pre-training)
  * Replaced during fine-tuning

- [PAD]: Padding token
  * Pads sequences to max length
  * Ignored in attention

ATTENTION MECHANISM:
====================
Multi-Head Self-Attention (same as Transformer encoder):

For each head h:
   Q = X @ W_Q^h    (Query)
   K = X @ W_K^h    (Key)
   V = X @ W_V^h    (Value)
   
   Attention(Q,K,V) = softmax(QK^T / √d_k) @ V

Multi-head concatenation:
   MultiHead = Concat(head_1, ..., head_h) @ W_O

Key properties:
- Bidirectional: No causal masking
- All tokens attend to all other tokens
- Attention weights show what tokens are relevant

Why multi-head:
- Different heads learn different patterns
- Some heads: syntactic relationships (subject-verb)
- Other heads: semantic relationships (co-reference)

FEED-FORWARD NETWORK:
=====================
Applied to each position independently:

   FFN(x) = GELU(x @ W_1 + b_1) @ W_2 + b_2

Where:
- W_1: (hidden_size, intermediate_size)
  * intermediate_size = 4 × hidden_size
  * BERT-Base: 768 → 3072
  
- W_2: (intermediate_size, hidden_size)
  * Projects back to original dimension
  * 3072 → 768

- GELU: Gaussian Error Linear Unit
  * Smoother than ReLU
  * GELU(x) = x × Φ(x)
  * Φ(x) = standard Gaussian CDF
  * Approximation: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

LAYER NORMALIZATION:
====================
Applied before attention and FFN (Pre-LN):

   x' = LayerNorm(x)
   x = x + SelfAttention(x')
   x' = LayerNorm(x)
   x = x + FFN(x')

Benefits:
- Stabilizes training
- Allows deeper networks
- Reduces internal covariate shift

Formula:
   LN(x) = γ × (x - μ) / √(σ² + ε) + β

Where:
- μ, σ: Mean and std over hidden dimensions
- γ, β: Learned scale and shift parameters
- ε: Small constant for numerical stability

FINE-TUNING:
============
After pre-training, BERT is fine-tuned on downstream tasks:

1. SEQUENCE CLASSIFICATION:
   - Sentiment analysis, spam detection
   - Use [CLS] token output
   - Add softmax layer on top

2. TOKEN CLASSIFICATION:
   - Named Entity Recognition (NER)
   - Part-of-Speech (POS) tagging
   - Use each token's output
   - Add classification layer per token

3. QUESTION ANSWERING:
   - SQuAD dataset
   - Input: [CLS] Question [SEP] Paragraph [SEP]
   - Predict start and end positions of answer
   - Two output layers: P_start and P_end

4. SENTENCE PAIR CLASSIFICATION:
   - Natural Language Inference (NLI)
   - Semantic similarity
   - Use [CLS] token output

Fine-tuning process:
- Initialize from pre-trained weights
- Add task-specific layer on top
- Train end-to-end on labeled data
- Very few epochs needed (2-4)
- Learning rate: 5e-5 to 2e-5

PRE-TRAINING DETAILS:
=====================
Dataset:
- BooksCorpus: 800M words
- English Wikipedia: 2,500M words
- Total: ~3.3 billion words

Training:
- Batch size: 256 sequences
- Steps: 1M (BERT-Base)
- Optimizer: Adam (β1=0.9, β2=0.999)
- Learning rate: 1e-4 with warmup
- Warmup: First 10,000 steps
- Dropout: 0.1 everywhere
- Training time: 4 days on 16 TPUs (BERT-Base)

Sequence preparation:
- Sample two spans from corpus
- 50% time: spans are consecutive (IsNext)
- 50% time: spans are random (NotNext)
- Combined length ≤ 512 tokens

BERT'S IMPACT:
==============
Performance gains over previous state-of-art:

GLUE Benchmark: +7.6% absolute
SQuAD v1.1 F1: 93.2 (human: 91.2!)
SQuAD v2.0 F1: 83.1
SWAG: 86.6%

Key innovations:
1. Bidirectional pre-training
2. Masked language modeling
3. Transfer learning paradigm
4. Simple fine-tuning procedure

Influence:
- Spawned many variants (RoBERTa, ALBERT, DistilBERT)
- Established transformer pre-training paradigm
- Foundation for modern NLP
- Led to GPT-3, T5, and beyond

ADVANTAGES:
===========
✓ Bidirectional context understanding
✓ Transfer learning: pre-train once, fine-tune many times
✓ State-of-art on 11 NLP tasks
✓ Relatively simple architecture
✓ Works well with limited labeled data

LIMITATIONS:
============
✗ Computationally expensive (340M params for BERT-Large)
✗ Static embeddings (can't handle new words well)
✗ NSP task may not be as useful (shown in RoBERTa)
✗ Mask tokens don't appear during fine-tuning (train-test mismatch)
✗ Fixed max sequence length (512 tokens)

KEY TAKEAWAYS:
==============
1. Bidirectional context is crucial for language understanding
2. Masked language modeling enables bidirectional training
3. Pre-training + fine-tuning is powerful paradigm
4. Simple architecture with clever training objectives
5. Transformer encoders excel at understanding tasks

Reference:
    Jacob Devlin et al. "BERT: Pre-training of Deep Bidirectional 
    Transformers for Language Understanding." arXiv:1810.04805, 2018.
    https://arxiv.org/abs/1810.04805
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class BERTEmbeddings(nn.Module):
    """
    BERT embeddings: Token + Segment + Position embeddings.
    
    The three embeddings are summed to create the final input representation.
    This allows BERT to understand:
    - What the token is (token embedding)
    - Which sentence it belongs to (segment embedding)
    - Where it is in the sequence (position embedding)
    
    Args:
        vocab_size: Size of vocabulary (30,522 for BERT)
        hidden_size: Hidden dimension (768 for BERT-Base)
        max_position_embeddings: Max sequence length (512)
        type_vocab_size: Number of segment types (2: sentence A and B)
        dropout: Dropout probability
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Token embeddings: Maps token IDs to vectors
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        
        # Position embeddings: Learned positional encoding
        # Unlike Transformer's sinusoidal, BERT learns position embeddings
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Segment embeddings: Differentiates sentence A from sentence B
        # 0 for tokens in first sentence, 1 for second sentence
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
        # Register position_ids as buffer (not a parameter)
        # These are just [0, 1, 2, ..., max_position_embeddings-1]
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1))
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_length) - Token IDs
            token_type_ids: (batch_size, seq_length) - Segment IDs (0 or 1)
            position_ids: (batch_size, seq_length) - Position IDs (optional)
        
        Returns:
            (batch_size, seq_length, hidden_size) - Combined embeddings
        """
        batch_size, seq_length = input_ids.shape
        
        # If position_ids not provided, use default [0, 1, 2, ...]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        # If token_type_ids not provided, assume all tokens are sentence A (0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Get three types of embeddings
        word_embeds = self.word_embeddings(input_ids)           # (B, L, H)
        position_embeds = self.position_embeddings(position_ids) # (B, L, H)
        token_type_embeds = self.token_type_embeddings(token_type_ids) # (B, L, H)
        
        # Sum all embeddings
        embeddings = word_embeds + position_embeds + token_type_embeds
        
        # Apply layer norm and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BERTSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for BERT.
    
    Key difference from GPT:
    - No causal masking (can attend to all positions)
    - Fully bidirectional attention
    
    Attention formula:
        Attention(Q, K, V) = softmax(QK^T / √d_k) @ V
    
    Args:
        hidden_size: Hidden dimension (768 for BERT-Base)
        num_attention_heads: Number of attention heads (12 for BERT-Base)
        dropout: Dropout probability
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert hidden_size % num_attention_heads == 0
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape and transpose for multi-head attention.
        
        Input: (batch_size, seq_length, all_head_size)
        Output: (batch_size, num_heads, seq_length, head_size)
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # new_shape: (batch_size, seq_length, num_heads, head_size)
        x = x.view(new_shape)                                                            # (batch_size, seq_length, num_heads, head_size)
        return x.permute(0, 2, 1, 3)                                                     # (batch_size, num_heads, seq_length, head_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, 1, 1, seq_length) - Optional mask
                           1 for real tokens, 0 for padding
        
        Returns:
            (batch_size, seq_length, hidden_size) - Attended output
        """
        # Project to Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))              # (batch_size, num_heads, seq_length, head_size)
        key_layer = self.transpose_for_scores(self.key(hidden_states))                  # (batch_size, num_heads, seq_length, head_size)
        value_layer = self.transpose_for_scores(self.value(hidden_states))              # (batch_size, num_heads, seq_length, head_size)
        
        # Compute attention scores: Q @ K^T / √d_k
        # (batch_size, num_heads, seq_length, head_size) @ (batch_size, num_heads, head_size, seq_length)
        # = (batch_size, num_heads, seq_length, seq_length)
        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        # Mask has 1 for real tokens, 0 for padding
        # We convert it to additive mask: 0 for real, -10000 for padding
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        # (batch_size, num_heads, seq_length, seq_length) @ (batch_size, num_heads, seq_length, head_size)
        # = (batch_size, num_heads, seq_length, head_size)
        context_layer = attention_probs @ value_layer
        
        # Reshape back to (batch_size, seq_length, hidden_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (batch_size, seq_length, num_heads, head_size)
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)  # new_shape: (batch_size, seq_length, hidden_size)
        context_layer = context_layer.view(new_shape)                  # (batch_size, seq_length, hidden_size)
        
        return context_layer


class BERTSelfOutput(nn.Module):
    """
    Output projection and residual connection for self-attention.
    
    Takes attention output and:
    1. Projects back to hidden_size
    2. Applies dropout
    3. Adds residual connection
    4. Applies layer normalization
    """
    def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Output from self-attention
            input_tensor: Original input (for residual connection)
        """
        hidden_states = self.dense(hidden_states)               
        hidden_states = self.dropout(hidden_states)             
        hidden_states = self.LayerNorm(hidden_states + input_tensor)    
        return hidden_states


class BERTAttention(nn.Module):
    """
    Complete attention module: Self-attention + output projection.
    
    This combines BERTSelfAttention and BERTSelfOutput.
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self = BERTSelfAttention(hidden_size, num_attention_heads, dropout)
        self.output = BERTSelfOutput(hidden_size, dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class BERTIntermediate(nn.Module):
    """
    Intermediate (first) layer of feed-forward network.
    
    Expands dimension from hidden_size to intermediate_size (4× larger).
    Uses GELU activation instead of ReLU.
    
    GELU (Gaussian Error Linear Unit):
        GELU(x) = x × Φ(x)
        where Φ(x) is the standard Gaussian CDF
    
    Approximation used:
        GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    """
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        # GELU is smoother than ReLU, works better in practice
        self.intermediate_act_fn = nn.GELU()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    """
    Output (second) layer of feed-forward network.
    
    Projects back from intermediate_size to hidden_size.
    Includes residual connection and layer normalization.
    """
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    """
    Single BERT transformer layer.
    
    Architecture:
        Input
          ↓
        LayerNorm → Self-Attention → Residual
          ↓
        LayerNorm → Feed-Forward → Residual
          ↓
        Output
    
    This is one "block" of BERT. BERT-Base has 12 of these.
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = BERTAttention(hidden_size, num_attention_heads, dropout)
        self.intermediate = BERTIntermediate(hidden_size, intermediate_size)
        self.output = BERTOutput(hidden_size, intermediate_size, dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attention_output = self.attention(hidden_states, attention_mask)
        
        # Feed-forward with residual
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        return layer_output


class BERTEncoder(nn.Module):
    """
    Stack of BERT layers.
    
    BERT-Base: 12 layers
    BERT-Large: 24 layers
    """
    def __init__(
        self,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            BERTLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, 1, 1, seq_length)
        
        Returns:
            (batch_size, seq_length, hidden_size)
        """
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        
        return hidden_states


class BERTPooler(nn.Module):
    """
    Pooler for [CLS] token representation.
    
    Takes the hidden state of [CLS] token (first token) and
    applies a dense layer + tanh activation.
    
    This representation is used for sequence-level tasks like
    classification and next sentence prediction.
    """
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
        
        Returns:
            (batch_size, hidden_size) - Pooled [CLS] representation
        """
        # Take hidden state of first token ([CLS])
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BERTModel(nn.Module):
    """
    BERT: Bidirectional Encoder Representations from Transformers.
    
    This is the base BERT model without task-specific heads.
    Returns both sequence output (all tokens) and pooled output ([CLS] token).
    
    For specific tasks, add task-specific layers on top:
    - Classification: Use pooled_output with softmax layer
    - Token classification: Use sequence_output with per-token classifier
    - Question Answering: Use sequence_output with start/end classifiers
    
    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        max_position_embeddings: Maximum sequence length
        type_vocab_size: Number of segment types
        dropout: Dropout probability
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embeddings = BERTEmbeddings(
            vocab_size,
            hidden_size,
            max_position_embeddings,
            type_vocab_size,
            dropout
        )
        
        self.encoder = BERTEncoder(
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout
        )
        
        self.pooler = BERTPooler(hidden_size)
        
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights following BERT paper.
        
        From the paper:
        - "We initialize all weights with a truncated normal distribution 
           with standard deviation of 0.02"
        - "All biases are initialized to 0"
        """
        def _init_module(module):
            if isinstance(module, nn.Linear):
                # Initialize with truncated normal (std=0.02)
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
        self.apply(_init_module)
    
    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert attention mask to format expected by attention layers.
        
        Input mask: (batch_size, seq_length)
            - 1 for real tokens
            - 0 for padding tokens
        
        Output mask: (batch_size, 1, 1, seq_length)
            - 0.0 for real tokens (no masking)
            - -10000.0 for padding (effectively -inf after softmax)
        """
        # Add dimensions: (B, L) → (B, 1, 1, L)
        extended_attention_mask = attention_mask[:, None, None, :]
        
        # Convert to float and create additive mask
        # 1.0 → 0.0 (no masking)
        # 0.0 → -10000.0 (mask out)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_length) - Token IDs
            attention_mask: (batch_size, seq_length) - 1 for real, 0 for padding
            token_type_ids: (batch_size, seq_length) - 0 for sent A, 1 for sent B
        
        Returns:
            sequence_output: (batch_size, seq_length, hidden_size)
                           - Contextualized representation for each token
            pooled_output: (batch_size, hidden_size)
                         - [CLS] token representation for sequence-level tasks
        """
        # Create default attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Convert attention mask to proper format
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        
        # Get embeddings
        embedding_output = self.embeddings(
            input_ids,
            token_type_ids=token_type_ids
        )
        
        # Pass through encoder layers
        sequence_output = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask
        )
        
        # Pool [CLS] token
        pooled_output = self.pooler(sequence_output)
        
        return sequence_output, pooled_output


class BERTForMaskedLM(nn.Module):
    """
    BERT with Masked Language Model head for pre-training.
    
    Predicts masked tokens given context. This is the MLM task.
    
    Example:
        Input:  [CLS] The cat is [MASK] [SEP]
        Target: [CLS] The cat is cute [SEP]
        Loss: Only on [MASK] position
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.bert = BERTModel(
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            type_vocab_size,
            dropout
        )
        
        # MLM head: Transform → LayerNorm → Linear to vocab
        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, vocab_size)
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            token_type_ids: (batch_size, seq_length)
            labels: (batch_size, seq_length) - Target token IDs (-100 for non-masked)
        
        Returns:
            prediction_scores: (batch_size, seq_length, vocab_size)
            loss: Scalar loss if labels provided
        """
        sequence_output, _ = self.bert(input_ids, attention_mask, token_type_ids) # (batch_size, seq_length, hidden_size)
        prediction_scores = self.cls(sequence_output)                             # (batch_size, seq_length, vocab_size)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding, ignored
            loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1)
            )
        
        return prediction_scores, loss


class BERTForSequenceClassification(nn.Module):
    """
    BERT for sequence classification tasks.
    
    Uses [CLS] token representation for classification.
    
    Tasks:
    - Sentiment analysis (positive/negative/neutral)
    - Spam detection (spam/not spam)
    - Entailment (entailment/contradiction/neutral)
    
    Args:
        num_labels: Number of classes
    """
    def __init__(
        self,
        num_labels: int = 2,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        self.bert = BERTModel(
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            type_vocab_size,
            dropout
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            token_type_ids: (batch_size, seq_length)
            labels: (batch_size,) - Class labels
        
        Returns:
            logits: (batch_size, num_labels)
            loss: Scalar loss if labels provided
        """
        _, pooled_output = self.bert(input_ids, attention_mask, token_type_ids)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return logits, loss


class BERTForTokenClassification(nn.Module):
    """
    BERT for token-level classification tasks.
    
    Classifies each token independently using its contextualized representation.
    
    Tasks:
    - Named Entity Recognition (NER): Person, Location, Organization, etc.
    - Part-of-Speech (POS) tagging: Noun, Verb, Adjective, etc.
    - Slot filling: Intent classification in dialogue
    
    Args:
        num_labels: Number of token classes
    """
    def __init__(
        self,
        num_labels: int = 9,  # For example: O, B-PER, I-PER, B-LOC, I-LOC, etc.
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        self.bert = BERTModel(
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            type_vocab_size,
            dropout
        )
        
        # Token classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            token_type_ids: (batch_size, seq_length)
            labels: (batch_size, seq_length) - Token labels
        
        Returns:
            logits: (batch_size, seq_length, num_labels)
            loss: Scalar loss if labels provided
        """
        sequence_output, _ = self.bert(input_ids, attention_mask, token_type_ids)
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only compute loss on non-padding tokens
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return logits, loss


class BERTForQuestionAnswering(nn.Module):
    """
    BERT for extractive question answering (SQuAD-style).
    
    Given a question and a paragraph, predict start and end positions
    of the answer span in the paragraph.
    
    Input format:
        [CLS] Question [SEP] Paragraph [SEP]
    
    Output:
        - start_logits: Probability distribution over start positions
        - end_logits: Probability distribution over end positions
    
    Example:
        Question: "Who wrote Harry Potter?"
        Paragraph: "Harry Potter is a series of novels written by J.K. Rowling..."
        Answer: "J.K. Rowling" (extract from paragraph)
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.bert = BERTModel(
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            type_vocab_size,
            dropout
        )
        
        # QA head: Predicts start and end positions
        # Output: (batch_size, seq_length, 2)
        # [:, :, 0] = start logits
        # [:, :, 1] = end logits
        self.qa_outputs = nn.Linear(hidden_size, 2)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            token_type_ids: (batch_size, seq_length)
            start_positions: (batch_size,) - Ground truth start positions
            end_positions: (batch_size,) - Ground truth end positions
        
        Returns:
            start_logits: (batch_size, seq_length)
            end_logits: (batch_size, seq_length)
            loss: Scalar loss if positions provided
        """
        sequence_output, _ = self.bert(input_ids, attention_mask, token_type_ids)
        
        # Get start and end logits
        logits = self.qa_outputs(sequence_output)  # (B, L, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (B, L)
        end_logits = end_logits.squeeze(-1)      # (B, L)
        
        loss = None
        if start_positions is not None and end_positions is not None:
            # Compute cross-entropy loss for both start and end
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
        
        return start_logits, end_logits, loss


# Helper function to create BERT variants
def bert_base(**kwargs):
    """
    BERT-Base configuration:
    - 12 layers
    - 768 hidden size
    - 12 attention heads
    - 110M parameters
    """
    config = {
        'vocab_size': 30522,
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'dropout': 0.1
    }
    config.update(kwargs)
    return BERTModel(**config)


def bert_large(**kwargs):
    """
    BERT-Large configuration:
    - 24 layers
    - 1024 hidden size
    - 16 attention heads
    - 340M parameters
    """
    config = {
        'vocab_size': 30522,
        'hidden_size': 1024,
        'num_hidden_layers': 24,
        'num_attention_heads': 16,
        'intermediate_size': 4096,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'dropout': 0.1
    }
    config.update(kwargs)
    return BERTModel(**config)


if __name__ == "__main__":
    # ========== DEMONSTRATION ==========
    print("=" * 70)
    print("BERT: Bidirectional Encoder Representations from Transformers")
    print("=" * 70)
    
    # Create BERT-Base model
    model = bert_base()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nBERT-Base Architecture:")
    print(f"  Layers: 12")
    print(f"  Hidden size: 768")
    print(f"  Attention heads: 12")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Example input
    batch_size = 2
    seq_length = 128
    
    # Simulate tokenized input:
    # [CLS] sentence A [SEP] sentence B [SEP]
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    token_type_ids = torch.cat([
        torch.zeros(batch_size, seq_length // 2, dtype=torch.long),  # Sentence A
        torch.ones(batch_size, seq_length // 2, dtype=torch.long)    # Sentence B
    ], dim=1)
    
    print(f"\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  token_type_ids: {token_type_ids.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        sequence_output, pooled_output = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    
    print(f"\nOutput shapes:")
    print(f"  sequence_output: {sequence_output.shape}")
    print(f"    ↳ Contextualized representation for each token")
    print(f"  pooled_output: {pooled_output.shape}")
    print(f"    ↳ [CLS] token representation for sequence-level tasks")
    
    # Demonstrate task-specific models
    print("\n" + "=" * 70)
    print("Task-Specific BERT Models")
    print("=" * 70)
    
    # 1. Sequence Classification
    print("\n1. Sequence Classification (Sentiment Analysis):")
    classifier = BERTForSequenceClassification(num_labels=3)  # Positive/Negative/Neutral
    with torch.no_grad():
        logits, _ = classifier(input_ids, attention_mask, token_type_ids)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Predictions: {torch.argmax(logits, dim=-1)}")
    
    # 2. Token Classification
    print("\n2. Token Classification (NER):")
    token_classifier = BERTForTokenClassification(num_labels=9)  # NER tags
    with torch.no_grad():
        logits, _ = token_classifier(input_ids, attention_mask, token_type_ids)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Per-token predictions shape: {torch.argmax(logits, dim=-1).shape}")
    
    # 3. Question Answering
    print("\n3. Question Answering (SQuAD):")
    qa_model = BERTForQuestionAnswering()
    with torch.no_grad():
        start_logits, end_logits, _ = qa_model(input_ids, attention_mask, token_type_ids)
    print(f"   Start logits shape: {start_logits.shape}")
    print(f"   End logits shape: {end_logits.shape}")
    print(f"   Predicted answer span: [{torch.argmax(start_logits[0])}, {torch.argmax(end_logits[0])}]")
    
    # 4. Masked Language Model
    print("\n4. Masked Language Model (Pre-training):")
    mlm_model = BERTForMaskedLM()
    with torch.no_grad():
        prediction_scores, _ = mlm_model(input_ids, attention_mask, token_type_ids)
    print(f"   Prediction scores shape: {prediction_scores.shape}")
    print(f"   Predicted tokens for position 10: {torch.argmax(prediction_scores[0, 10])}")
    
    print("\n" + "=" * 70)
    print("Key Features:")
    print("=" * 70)
    print("✓ Bidirectional context understanding")
    print("✓ Pre-training with MLM + NSP")
    print("✓ Fine-tuning for various downstream tasks")
    print("✓ State-of-art performance on GLUE, SQuAD, etc.")
    print("✓ Transfer learning: pre-train once, fine-tune many times")
    
    print("\n" + "=" * 70)
    print("Comparison with GPT:")
    print("=" * 70)
    print("BERT (Encoder-only):")
    print("  - Bidirectional attention")
    print("  - Best for understanding tasks (classification, QA, NER)")
    print("  - Masked language modeling")
    
    print("\nGPT (Decoder-only):")
    print("  - Unidirectional (causal) attention")
    print("  - Best for generation tasks (text completion, dialogue)")
    print("  - Autoregressive language modeling")
    
    print("\n" + "=" * 70)

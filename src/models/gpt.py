"""
TinyGPT model with configurable attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..attention import ATTENTION_CLASSES, NOVEL_ACTIVATION_TYPES, FastAttention
from .blocks import GPT2Block, NovelBlock


class TinyGPT(nn.Module):
    """GPT-style language model with configurable attention."""
    
    def __init__(self, vocab_size, config, attention_type='performer', freeze_embeddings=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention_type = attention_type
        self.freeze_embeddings = freeze_embeddings
        self.config = config
        
        # Extract config values
        embed_dim = config['embed_dim']
        context_len = config['context_len']
        n_layers = config['n_layers']
        n_heads = config['n_heads']
        use_triton = config.get('use_triton', False)
        
        # Get attention class from registry
        attention_class = ATTENTION_CLASSES.get(attention_type, FastAttention)
        
        # Use NovelBlock for Yat variants, GPT2Block for others
        # if attention_type in NOVEL_ACTIVATION_TYPES:
        #     block_class = NovelBlock
        # else:
        #     block_class = GPT2Block
        block_class = GPT2Block
        
        dropout = config.get('dropout', 0.1)
        
        # Collect attention-specific kwargs from config
        attn_kwargs = {
            'num_rff_features': config.get('num_rff_features'),
            'num_prf_features': config.get('num_prf_features'),
            'num_poly_features': config.get('num_poly_features'),
            'num_quadrature_nodes': config.get('num_quadrature_nodes'),
            'epsilon': config.get('epsilon'),
        }
        # Filter out None values just in case
        attn_kwargs = {k: v for k, v in attn_kwargs.items() if v is not None}
        
        self.tok = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(context_len, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)  # Embedding dropout
        self.blocks = nn.ModuleList([
            block_class(embed_dim, n_heads, attention_class, use_triton=use_triton, dropout=dropout, **attn_kwargs) 
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Tie weights: output layer shares embedding weights
        self.head.weight = self.tok.weight
        
        # Initialize weights first
        self.apply(self._init_weights)
        
        # Zero the final LayerNorm bias (GPT-2 style)
        with torch.no_grad():
            self.ln.bias.zero_()
        
        # Then optionally load pretrained embeddings
        if freeze_embeddings:
            self._load_pretrained_embeddings()
    
    def _load_pretrained_embeddings(self):
        """Load pretrained GPT-2 embeddings and freeze them."""
        from transformers import GPT2LMHeadModel
        print("Loading pretrained GPT-2 embeddings...")
        
        embed_dim = self.config['embed_dim']
        
        gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        pretrained_embeds = gpt2.transformer.wte.weight.data  # (50257, 768)
        
        # Handle dimension mismatch
        if pretrained_embeds.shape[1] != embed_dim:
            print(f"Projecting embeddings from {pretrained_embeds.shape[1]} to {embed_dim}...")
            # Use PCA-style projection (take first embed_dim dims or pad)
            if embed_dim < pretrained_embeds.shape[1]:
                pretrained_embeds = pretrained_embeds[:, :embed_dim]
            else:
                # Pad with zeros
                padding = torch.zeros(pretrained_embeds.shape[0], embed_dim - pretrained_embeds.shape[1])
                pretrained_embeds = torch.cat([pretrained_embeds, padding], dim=1)
        
        # Copy weights
        with torch.no_grad():
            self.tok.weight.copy_(pretrained_embeds[:self.vocab_size])
        
        # Freeze embedding layer (and thus the tied output layer)
        self.tok.weight.requires_grad = False
        print(f"Embeddings frozen. Shape: {self.tok.weight.shape}")
        
        # Clean up
        del gpt2
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None):
        x = x.long()
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.embed_dropout(self.tok(x) + self.pos(pos))  # Apply embedding dropout
        
        # Gradient Checkpointing Loop
        for block in self.blocks:
            if self.training:
                # use_reentrant=False is standard for modern PyTorch
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        # if self.attention_type in NOVEL_ACTIVATION_TYPES:
        #     pass
        # else:
        #     x = self.ln(x)
        x = self.ln(x)
                

        logits = self.head(x)
        if y is None:
            return logits
        return F.cross_entropy(
            logits.float().view(-1, self.vocab_size),  # ✅ force fp32
            y.long().view(-1)
        )

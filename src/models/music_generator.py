"""Modern music generation models using PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class MusicTransformer(nn.Module):
    """Transformer-based music generation model."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_length: int = 1024
    ):
        """Initialize transformer model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            max_length: Maximum sequence length
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Create causal mask for autoregressive generation
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)
        
        # Transformer forward pass
        x = self.transformer(x, mask=causal_mask)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 500,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> torch.Tensor:
        """Generate music sequence.
        
        Args:
            input_ids: Initial sequence of shape (batch_size, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Penalty for repetition
            
        Returns:
            Generated sequence of shape (batch_size, generated_length)
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for the last position
                logits = self.forward(generated)[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            if logits[i, token_id] < 0:
                                logits[i, token_id] *= repetition_penalty
                            else:
                                logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end token (assuming token ID 2 is END)
                if (next_token == 2).all():
                    break
                    
        return generated


class MusicLSTM(nn.Module):
    """LSTM-based music generation model."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.3
    ):
        """Initialize LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            n_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, vocab_size)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            hidden: Hidden state tuple (h_0, c_0)
            
        Returns:
            Tuple of (output_logits, new_hidden_state)
        """
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Output projection
        out = self.dropout(lstm_out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits, hidden
        
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device
            
        Returns:
            Initial hidden state tuple
        """
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return h_0, c_0
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 500,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Generate music sequence.
        
        Args:
            input_ids: Initial sequence of shape (batch_size, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            
        Returns:
            Generated sequence of shape (batch_size, generated_length)
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        generated = input_ids.clone()
        hidden = self.init_hidden(batch_size, device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for the last position
                logits, hidden = self.forward(generated[:, -1:], hidden)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end token
                if (next_token == 2).all():
                    break
                    
        return generated


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """Create model based on configuration.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    model_name = model_config.get('name', 'transformer')
    vocab_size = model_config.get('vocab_size', 128)
    
    if model_name == 'transformer':
        return MusicTransformer(
            vocab_size=vocab_size,
            d_model=model_config.get('d_model', 512),
            n_heads=model_config.get('n_heads', 8),
            n_layers=model_config.get('n_layers', 6),
            dropout=model_config.get('dropout', 0.1),
            max_length=model_config.get('max_length', 1024)
        )
    elif model_name == 'lstm':
        return MusicLSTM(
            vocab_size=vocab_size,
            embedding_dim=model_config.get('embedding_dim', 256),
            hidden_dim=model_config.get('hidden_dim', 512),
            n_layers=model_config.get('n_layers', 3),
            dropout=model_config.get('dropout', 0.3)
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")

"""
RibonanzaNet: Neural network architecture for RNA reactivity and secondary structure prediction.

This module implements the RibonanzaNet model, which uses transformer-based architecture with
pairwise features and triangular attention mechanisms for RNA sequence analysis. The model can
predict both RNA reactivity values and secondary structure (2D structure in dot-bracket notation).

Key Components:
    - RibonanzaNet: Base model for RNA reactivity prediction
    - RibonanzaNetSS: Extended model for secondary structure prediction
    - ConvTransformerEncoderLayer: Transformer layers with pairwise features
    - TriangleAttention: Attention mechanism for pairwise representations
    - TriangleMultiplicativeModule: Updates for pairwise features

Note: RibonanzaNet uses different tokenization than gRNAde (ACGU vs ACGUN).
"""

import math
import os
from typing import Optional, Union, List, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import einsum
from einops import rearrange


# ============================================================================
# Configuration and Utilities
# ============================================================================

class Config:
    """Configuration container for model hyperparameters.
    
    Dynamically creates attributes from keyword arguments, providing a flexible
    way to store and access model configuration parameters.
    
    Args:
        **entries: Arbitrary keyword arguments representing configuration parameters.
    """
    
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        """Print all configuration entries."""
        print(self.entries)


def mask_diagonal(matrix: np.ndarray, mask_value: float = 0) -> np.ndarray:
    """Mask the diagonal and near-diagonal elements of a matrix.
    
    Sets elements where |i - j| < 4 to mask_value. This is used to prevent
    base pairs that are too close in sequence for RNA secondary structure prediction.
    
    Args:
        matrix: Square matrix to mask.
        mask_value: Value to use for masked elements. Default is 0.
        
    Returns:
        Masked copy of the input matrix.
    """
    matrix = matrix.copy()
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < 4:
                matrix[i][j] = mask_value
    return matrix


# ============================================================================
# Main Model Classes
# ============================================================================

class RibonanzaNet(nn.Module):
    """Transformer-based model for RNA reactivity prediction.
    
    RibonanzaNet uses a multi-layer transformer architecture with pairwise features,
    triangular attention, and outer product updates to predict RNA reactivity values
    from nucleotide sequences. The model incorporates both sequence-level and pairwise
    representations for enhanced prediction accuracy.
    
    Architecture:
        - Embedding layer for nucleotide sequences
        - Multiple ConvTransformerEncoderLayers with pairwise feature updates
        - Triangular multiplicative updates and attention mechanisms
        - Linear decoder for reactivity prediction
    
    Note: Uses ACGU tokenization (different from gRNAde's ACGUN).
    """

    def __init__(
        self,
        config_filepath: str = "config.yaml",
        checkpoint_filepath: Optional[str] = "ribonanzanet.pt",
        device: str = "cpu",
    ):
        """Initialize RibonanzaNet model.
        
        Loads model configuration from YAML file and optionally loads pretrained weights.
        Sets up transformer encoder layers, embeddings, and prediction heads.

        Args:
            config_filepath: Path to the YAML configuration file containing model hyperparameters.
            checkpoint_filepath: Path to the checkpoint file with pretrained weights.
                Set to None to initialize without loading weights.
            device: Device on which to run the model ('cpu', 'cuda', etc.).
        """

        super(RibonanzaNet, self).__init__()

        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)
        config = Config(**config)
        self.config = config

        # Note: RibonanzaNet uses a different tokenisation than gRNAde!
        self.tokens = {nt: i for i, nt in enumerate("ACGU")}

        nhid = config.ninp * 4

        self.transformer_encoder = []
        for i in range(config.nlayers):
            if i != config.nlayers - 1:
                k = config.k
            else:
                k = 1
            self.transformer_encoder.append(
                ConvTransformerEncoderLayer(
                    d_model=config.ninp,
                    nhead=config.nhead,
                    dim_feedforward=nhid,
                    pairwise_dimension=config.pairwise_dimension,
                    use_triangular_attention=config.use_triangular_attention,
                    dropout=config.dropout,
                    k=k,
                )
            )
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        self.encoder = nn.Embedding(config.ntoken, config.ninp, padding_idx=4)
        self.decoder = nn.Linear(config.ninp, config.nclass)

        self.outer_product_mean = Outer_Product_Mean(
            in_dim=config.ninp, pairwise_dim=config.pairwise_dimension
        )
        self.pos_encoder = RelativePositionalEncoding(config.pairwise_dimension)

        if checkpoint_filepath is not None:
            print(f"Loading RibonanzaNet checkpoint: {checkpoint_filepath}")
            self.load_state_dict(torch.load(checkpoint_filepath, map_location="cpu"))

        self.device = device

    @torch.no_grad()
    def predict(self, sequence: Union[str, List[str]]) -> torch.Tensor:
        """Predict RNA reactivity for single or multiple sequences.
        
        This is the main inference method. Automatically handles tokenization,
        batching, and device management. No gradients are computed.

        Args:
            sequence: RNA sequence(s) as string(s) containing only ACGU characters.
                Can be a single string or list of strings for batch prediction.
        
        Returns:
            Predicted reactivity values as a tensor:
                - Shape (L,) for single sequence input
                - Shape (B, L) for batch input
            where L is sequence length and B is batch size.
            
        Example:
            >>> model = RibonanzaNet()
            >>> reactivity = model.predict("ACGUACGU")
            >>> batch_reactivity = model.predict(["ACGU", "UGCA"])
        """
        if isinstance(sequence, str):
            # Single sequence
            seq_tokenized = (
                torch.tensor([self.tokens[letter] for letter in sequence])
                .int()
                .unsqueeze(0)
                .to(self.device)
            )
            mask = torch.ones_like(seq_tokenized)  # no masking
            preds = self.forward(seq_tokenized, mask).squeeze(0).cpu()
        else:
            # Batch of sequences
            seq_tokenized = (
                torch.tensor(
                    [[self.tokens[letter] for letter in seq] for seq in sequence]
                )
                .int()
                .to(self.device)
            )
            mask = torch.ones_like(seq_tokenized)  # no masking
            preds = self.forward(seq_tokenized, mask).cpu()
        return preds

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_aw: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """Forward pass through the RibonanzaNet model.
        
        Processes tokenized RNA sequences through embedding, transformer layers with
        pairwise features, and a linear decoder to predict reactivity values.
        
        Args:
            src: Tokenized input sequences of shape (B, L) where B is batch size
                and L is sequence length. Values should be integers in [0, 3] for ACGU.
            src_mask: Optional mask for padding tokens, shape (B, L). 1 for valid tokens,
                0 for padding. If None, no masking is applied.
            return_aw: If True, also return attention weights from all layers.
                
        Returns:
            If return_aw is False:
                Predicted reactivity values of shape (B, L).
            If return_aw is True:
                Tuple of (predictions, attention_weights_list).
        """
        B, L = src.shape
        src = src
        src = self.encoder(src).reshape(B, L, -1)

        pairwise_features = self.outer_product_mean(src)
        pairwise_features = pairwise_features + self.pos_encoder(src)

        attention_weights = []
        for i, layer in enumerate(self.transformer_encoder):
            if src_mask is not None:
                if return_aw:
                    src, aw = layer(
                        src, pairwise_features, src_mask, return_aw=return_aw
                    )
                    attention_weights.append(aw)
                else:
                    src, pairwise_features = layer(
                        src, pairwise_features, src_mask, return_aw=return_aw
                    )
            else:
                if return_aw:
                    src, aw = layer(src, pairwise_features, return_aw=return_aw)
                    attention_weights.append(aw)
                else:
                    src, pairwise_features = layer(
                        src, pairwise_features, return_aw=return_aw
                    )

        # Save pairwise features for 2D structure prediction
        # (used by RibonanzaNetSS subclass)
        self.pairwise_features = pairwise_features
        
        # Decode to final predictions
        # Note: pairwise_features.mean() * 0 is a gradient trick for proper backprop
        output = self.decoder(src).squeeze(-1) + pairwise_features.mean() * 0

        if return_aw:
            return output, attention_weights
        else:
            return output


class RibonanzaNetSS(RibonanzaNet):
    """Extended RibonanzaNet model for RNA secondary structure prediction.
    
    Inherits from RibonanzaNet and adds a secondary structure prediction head that
    operates on the pairwise features to predict base pairing probabilities.
    Uses the Hungarian algorithm for optimal base pair assignment.
    
    The model predicts both reactivity (from parent class) and 2D structure in
    dot-bracket notation.
    """

    def __init__(
        self,
        config_filepath: str = "config.yaml",
        checkpoint_filepath: Optional[str] = "ribonanzanet_ss.pt",
        device: str = "cpu",
    ):
        """Initialize RibonanzaNet SS model for secondary structure prediction.
        
        Loads the base RibonanzaNet architecture and adds a contact prediction head.
        Also sets up Arnie configuration for structure post-processing.

        Args:
            config_filepath: Path to the YAML configuration file.
            checkpoint_filepath: Path to checkpoint with pretrained SS weights.
                Set to None to initialize without loading weights.
            device: Device on which to run the model ('cpu', 'cuda', etc.).
        """

        super(RibonanzaNetSS, self).__init__(config_filepath, None, device)
        self.dropout = nn.Dropout(0.0)
        self.ct_predictor = nn.Linear(64, 1)  # Contact prediction head

        if checkpoint_filepath is not None:
            print(f"Loading RibonanzaNet SS checkpoint: {checkpoint_filepath}")
            self.load_state_dict(torch.load(checkpoint_filepath, map_location="cpu"))

        self.device = device

        # Create dummy Arnie config for structure post-processing
        # Arnie is used for Hungarian algorithm-based structure parsing
        with open('arnie_file.txt', 'w+') as f:
            f.write("linearpartition: . \nTMP: /tmp")    
        os.environ['ARNIEFILE'] = 'arnie_file.txt'

    @torch.no_grad()
    def predict(
        self,
        sequence: Union[str, List[str], torch.Tensor]
    ) -> Tuple[np.ndarray, List[str]]:
        """Predict RNA secondary structure for single or multiple sequences.
        
        This method predicts base pairing probabilities and converts them to
        discrete secondary structures using the Hungarian algorithm. The diagonal
        and near-diagonal (|i-j| < 4) are masked to enforce minimum loop size.

        Args:
            sequence: Input can be:
                - str: Single RNA sequence (ACGU)
                - List[str]: Batch of RNA sequences
                - torch.Tensor: Pre-tokenized sequences of shape (B, L)
        
        Returns:
            Tuple containing:
                - preds: Base pairing probability matrix of shape (B, L, L).
                    Values range from 0 to 1, where higher values indicate
                    higher probability of base pairing between positions i and j.
                - hungarian_structures: List of secondary structures in dot-bracket
                    notation, where '.' represents unpaired bases and '(' ')'  
                    represent paired bases.
                    
        Example:
            >>> model = RibonanzaNetSS()
            >>> probs, structures = model.predict("ACGUACGU")
            >>> print(structures[0])  # e.g., "((...))."
        """
        if isinstance(sequence, str):
            # Single sequence
            seq_tokenized = (
                torch.tensor([self.tokens[letter] for letter in sequence])
                .int()
                .unsqueeze(0)
                .to(self.device)
            )
            mask = torch.ones_like(seq_tokenized)  # no masking
            self.forward(seq_tokenized, mask)

        elif isinstance(sequence, torch.Tensor):
            # Already tokenized sequence
            mask = torch.ones_like(sequence)  # no masking
            self.forward(sequence, mask)
        
        else:
            # Batch of sequences
            seq_tokenized = (
                torch.tensor(
                    [[self.tokens[letter] for letter in seq] for seq in sequence]
                )
                .int()
                .to(self.device)
            )
            mask = torch.ones_like(seq_tokenized)  # no masking
            self.forward(seq_tokenized, mask)
            
        # Extract pairwise features from the last forward pass
        pairwise_features = self.pairwise_features
        
        # Symmetrize pairwise features (ensure P[i,j] == P[j,i])
        pairwise_features = pairwise_features + pairwise_features.permute(0, 2, 1, 3)
        
        # Predict base pairing probabilities
        preds = self.ct_predictor(
            self.dropout(pairwise_features)
        ).sigmoid().squeeze(-1).cpu().numpy()  # Shape: (B, L, L)

        # Convert probabilities to discrete structures using Hungarian algorithm
        from arnie.pk_predictors import _hungarian
        test_preds_hungarian = []
        hungarian_structures = []
        hungarian_bps = []
        
        for i in range(len(preds)):
            # Apply Hungarian algorithm with theta=0.5 (optimal from validation)
            # Mask diagonal to enforce minimum loop size of 4
            s, bp = _hungarian(mask_diagonal(preds[i]), theta=0.5, min_len_helix=1)
            hungarian_bps.append(bp)
            
            # Convert base pairs to contact matrix
            ct_matrix = np.zeros((len(s), len(s)))
            for b in bp:
                ct_matrix[b[0], b[1]] = 1
            ct_matrix = ct_matrix + ct_matrix.T  # Make symmetric
            test_preds_hungarian.append(ct_matrix)
            hungarian_structures.append(s)
        
        return preds, hungarian_structures


# ============================================================================
# Attention and Transformer Components
# ============================================================================

class TriangleAttention(nn.Module):
    """Triangle attention mechanism for pairwise representations.
    
    Implements axial attention over rows or columns of pairwise feature matrices,
    as described in AlphaFold2. This allows the model to update pairwise features
    by attending along one axis while maintaining information about the other.
    
    Args:
        in_dim: Input dimension of pairwise features.
        dim: Dimension per attention head.
        n_heads: Number of attention heads.
        wise: Direction of attention, either "row" or "col" for row-wise or column-wise.
    """
    
    def __init__(self, in_dim: int = 128, dim: int = 32, n_heads: int = 4, wise: str = "row"):
        super(TriangleAttention, self).__init__()
        self.n_heads = n_heads
        self.wise = wise
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.linear_for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Sigmoid())
        self.to_out = nn.Linear(n_heads * dim, in_dim)

    def forward(self, z: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Apply triangle attention to pairwise features.
        
        Args:
            z: Pairwise features of shape (B, L, L, D).
            src_mask: Sequence mask of shape (B, L) with 1 for valid positions, 0 for padding.
            
        Returns:
            Updated pairwise features of shape (B, L, L, D).
        
        Masking strategy:
        For row triangular attention:
        - The attention matrix is brijh, where b is the batch, r is the row, and h is the head.
        - To create the mask, take the self-attention mask and unsqueeze it along dimensions 1 and -1.
        - Add negative infinity to the matrix before applying softmax.

        For column triangular attention:
        - The attention matrix is bijlh.
        - To create the mask, take the self-attention mask and unsqueeze it along dimensions 3 and -1.

        To create the pairwise mask:
        - Take the src_mask and generate the pairwise mask.
        - Unsqueeze the pairwise mask accordingly.
        """
        src_mask[src_mask == 0] = -1
        src_mask = src_mask.unsqueeze(-1).float()
        attn_mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1))

        wise = self.wise
        z = self.norm(z)
        q, k, v = torch.chunk(self.to_qkv(z), 3, -1)
        q, k, v = map(
            lambda x: rearrange(x, "b i j (h d)->b i j h d", h=self.n_heads), (q, k, v)
        )
        b = self.linear_for_pair(z)
        gate = self.to_gate(z)
        scale = q.size(-1) ** 0.5
        if wise == "row":
            eq_attn = "brihd,brjhd->brijh"
            eq_multi = "brijh,brjhd->brihd"
            b = rearrange(b, "b i j (r h)->b r i j h", r=1)
            softmax_dim = 3
            attn_mask = rearrange(attn_mask, "b i j->b 1 i j 1")
        elif wise == "col":
            eq_attn = "bilhd,bjlhd->bijlh"
            eq_multi = "bijlh,bjlhd->bilhd"
            b = rearrange(b, "b i j (l h)->b i j l h", l=1)
            softmax_dim = 2
            attn_mask = rearrange(attn_mask, "b i j->b i j 1 1")
        else:
            raise ValueError("wise should be col or row!")

        logits = torch.einsum(eq_attn, q, k) / scale + b
        logits = logits.masked_fill(attn_mask == -1, float("-1e-9"))
        attn = logits.softmax(softmax_dim)

        out = torch.einsum(eq_multi, attn, v)
        out = gate * rearrange(out, "b i j h d-> b i j (h d)")
        z_ = self.to_out(out)
        return z_


class TriangleMultiplicativeModule(nn.Module):
    """Triangle multiplicative update for pairwise features.
    
    Implements the triangle multiplicative update from AlphaFold2, which updates
    pairwise features by combining information from two edges of a triangle to
    update the third edge. This helps propagate information through the pairwise
    representation graph.
    
    Args:
        dim: Dimension of pairwise features.
        hidden_dim: Hidden dimension for projections. Defaults to dim if not specified.
        mix: Update direction, either "ingoing" or "outgoing" for different edge combinations.
    """
    
    def __init__(self, *, dim: int, hidden_dim: Optional[int] = None, mix: str = "ingoing"):
        super().__init__()
        assert mix in {"ingoing", "outgoing"}, "mix must be either ingoing or outgoing"

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.0)
            nn.init.constant_(gate.bias, 1.0)

        if mix == "outgoing":
            self.mix_einsum_eq = "... i k d, ... j k d -> ... i j d"
        elif mix == "ingoing":
            self.mix_einsum_eq = "... k j d, ... k i d -> ... i j d"

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply triangle multiplicative update.
        
        Args:
            x: Pairwise features of shape (B, L, L, D).
            src_mask: Optional sequence mask of shape (B, L).
            
        Returns:
            Updated pairwise features of shape (B, L, L, D).
        """
        src_mask = src_mask.unsqueeze(-1).float()
        mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1))
        assert x.shape[1] == x.shape[2], "feature map must be symmetrical"
        if exists(mask):
            mask = rearrange(mask, "b i j -> b i j ()")

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class ConvTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with convolutional and pairwise feature updates.
    
    This layer combines:
        - 1D convolution for local context
        - Multi-head self-attention with pairwise bias
        - Feedforward network
        - Pairwise feature updates via outer product and triangle operations
        - Optional triangle attention mechanisms
    
    This architecture is inspired by AlphaFold2's Evoformer blocks.
    
    Args:
        d_model: Dimension of sequence representations.
        nhead: Number of attention heads.
        dim_feedforward: Hidden dimension of feedforward network.
        pairwise_dimension: Dimension of pairwise features.
        use_triangular_attention: Whether to use triangle attention in addition to
            triangle multiplicative updates.
        dropout: Dropout rate.
        k: Kernel size for 1D convolution.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        pairwise_dimension: int,
        use_triangular_attention: bool,
        dropout: float = 0.1,
        k: int = 3,
    ):
        super(ConvTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            d_model, nhead, d_model // nhead, d_model // nhead, dropout=dropout
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.pairwise2heads = nn.Linear(pairwise_dimension, nhead, bias=False)
        self.pairwise_norm = nn.LayerNorm(pairwise_dimension)
        self.activation = nn.GELU()

        self.conv = nn.Conv1d(d_model, d_model, k, padding=k // 2)

        self.triangle_update_out = TriangleMultiplicativeModule(
            dim=pairwise_dimension, mix="outgoing"
        )
        self.triangle_update_in = TriangleMultiplicativeModule(
            dim=pairwise_dimension, mix="ingoing"
        )

        self.pair_dropout_out = DropoutRowwise(dropout)
        self.pair_dropout_in = DropoutRowwise(dropout)

        self.use_triangular_attention = use_triangular_attention
        if self.use_triangular_attention:
            self.triangle_attention_out = TriangleAttention(
                in_dim=pairwise_dimension, dim=pairwise_dimension // 4, wise="row"
            )
            self.triangle_attention_in = TriangleAttention(
                in_dim=pairwise_dimension, dim=pairwise_dimension // 4, wise="col"
            )

            self.pair_attention_dropout_out = DropoutRowwise(dropout)
            self.pair_attention_dropout_in = DropoutColumnwise(dropout)

        self.outer_product_mean = Outer_Product_Mean(
            in_dim=d_model, pairwise_dim=pairwise_dimension
        )

        self.pair_transition = nn.Sequential(
            nn.LayerNorm(pairwise_dimension),
            nn.Linear(pairwise_dimension, pairwise_dimension * 4),
            nn.ReLU(inplace=True),
            nn.Linear(pairwise_dimension * 4, pairwise_dimension),
        )

    def forward(
        self,
        src: torch.Tensor,
        pairwise_features: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_aw: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass through the transformer encoder layer.
        
        Args:
            src: Sequence representations of shape (B, L, D).
            pairwise_features: Pairwise features of shape (B, L, L, P).
            src_mask: Optional sequence mask of shape (B, L).
            return_aw: If True, also return attention weights.
            
        Returns:
            If return_aw is False:
                Tuple of (updated_src, updated_pairwise_features).
            If return_aw is True:
                Tuple of (updated_src, updated_pairwise_features, attention_weights).
        """

        src = src * src_mask.float().unsqueeze(-1)

        res = src
        src = src + self.conv(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.norm3(src)

        pairwise_bias = self.pairwise2heads(
            self.pairwise_norm(pairwise_features)
        ).permute(0, 3, 1, 2)
        src2, attention_weights = self.self_attn(
            src, src, src, mask=pairwise_bias, src_mask=src_mask
        )

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        pairwise_features = pairwise_features + self.outer_product_mean(src)
        pairwise_features = pairwise_features + self.pair_dropout_out(
            self.triangle_update_out(pairwise_features, src_mask)
        )
        pairwise_features = pairwise_features + self.pair_dropout_in(
            self.triangle_update_in(pairwise_features, src_mask)
        )

        if self.use_triangular_attention:
            pairwise_features = pairwise_features + self.pair_attention_dropout_out(
                self.triangle_attention_out(pairwise_features, src_mask)
            )
            pairwise_features = pairwise_features + self.pair_attention_dropout_in(
                self.triangle_attention_in(pairwise_features, src_mask)
            )

        pairwise_features = pairwise_features + self.pair_transition(pairwise_features)

        if return_aw:
            return src, pairwise_features, attention_weights
        else:
            return src, pairwise_features


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional pairwise bias.
    
    Standard multi-head attention mechanism with support for additive pairwise bias.
    The pairwise bias allows incorporating structural information (e.g., from pairwise
    features) into the attention computation.
    
    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        d_k: Dimension per head for keys and queries.
        d_v: Dimension per head for values.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_head: int, d_k: int, d_v: int, dropout: float = 0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head attention.
        
        Args:
            q: Query tensor of shape (B, L, D).
            k: Key tensor of shape (B, L, D).
            v: Value tensor of shape (B, L, D).
            mask: Optional pairwise bias of shape (B, n_head, L, L) to add to attention logits.
            src_mask: Optional sequence mask of shape (B, L) for padding.
            
        Returns:
            Tuple of (output, attention_weights) where output has shape (B, L, D) and
            attention_weights has shape (B, n_head, L, L).
        """

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask  # For head axis broadcasting

        if src_mask is not None:
            src_mask[src_mask == 0] = -1
            src_mask = src_mask.unsqueeze(-1).float()
            attn_mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1)).unsqueeze(1)
            q, attn = self.attention(q, k, v, mask=mask, attn_mask=attn_mask)
        else:
            q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism.
    
    Computes attention weights as softmax(QK^T / sqrt(d_k)) and applies them to values.
    Supports additive bias and masking.
    
    Args:
        temperature: Scaling factor (typically sqrt(d_k)).
        attn_dropout: Dropout rate applied to attention weights.
    """

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention.
        
        Args:
            q: Query tensor of shape (B, n_head, L, d_k).
            k: Key tensor of shape (B, n_head, L, d_k).
            v: Value tensor of shape (B, n_head, L, d_v).
            mask: Optional additive bias of shape (B, n_head, L, L).
            attn_mask: Optional multiplicative mask (1 for valid, -1 for invalid).
            
        Returns:
            Tuple of (output, attention_weights) where output has shape (B, n_head, L, d_v)
            and attention_weights has shape (B, n_head, L, L).
        """

        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature

        if mask is not None:
            attn = attn + mask  # this is actually the bias

        if attn_mask is not None:
            attn = attn.float().masked_fill(attn_mask == -1, float("-1e-9"))

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)

        return output, attn


# ============================================================================
# Positional Encoding and Feature Modules
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.
    
    Adds position-dependent sinusoidal embeddings to sequence representations,
    allowing the model to use sequence position information.
    
    Args:
        d_model: Model dimension.
        dropout: Dropout rate.
        max_len: Maximum sequence length to precompute encodings for.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (L, B, D) or (B, L, D).
            
        Returns:
            Input with positional encoding added, same shape as input.
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Outer_Product_Mean(nn.Module):
    """Outer product update for pairwise features.
    
    Computes an outer product between projected sequence representations to update
    pairwise features. This operation creates pairwise information from single-sequence
    representations, similar to AlphaFold2's outer product mean.
    
    Args:
        in_dim: Input dimension of sequence representations.
        dim_msa: Intermediate dimension after first projection.
        pairwise_dim: Output dimension of pairwise features.
    """
    
    def __init__(self, in_dim: int = 256, dim_msa: int = 32, pairwise_dim: int = 64):
        super(Outer_Product_Mean, self).__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)
        self.proj_down2 = nn.Linear(dim_msa**2, pairwise_dim)

    def forward(
        self,
        seq_rep: torch.Tensor,
        pair_rep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute outer product of sequence representations.
        
        Args:
            seq_rep: Sequence representations of shape (B, L, D).
            pair_rep: Optional existing pairwise features to add to. Shape (B, L, L, P).
            
        Returns:
            Pairwise features of shape (B, L, L, P). If pair_rep is provided,
            the outer product is added to it; otherwise returns just the outer product.
        """
        seq_rep = self.proj_down1(seq_rep)
        outer_product = torch.einsum("bid,bjc -> bijcd", seq_rep, seq_rep)
        outer_product = rearrange(outer_product, "b i j c d -> b i j (c d)")
        outer_product = self.proj_down2(outer_product)

        if pair_rep is not None:
            outer_product = outer_product + pair_rep

        return outer_product


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for pairwise features.
    
    Encodes the relative distance between sequence positions as one-hot vectors,
    which are then projected to a learned embedding. Distances are clipped to [-8, 8]
    to limit the vocabulary size.
    
    Args:
        dim: Output dimension of the positional encoding.
    """

    def __init__(self, dim: int = 64):
        super(RelativePositionalEncoding, self).__init__()
        self.linear = nn.Linear(17, dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encodings.
        
        Args:
            src: Input tensor of shape (B, L, D) used to infer batch size and length.
            
        Returns:
            Relative positional encodings of shape (L, L, dim) representing the
            learned embedding of relative distances between all position pairs.
        """
        L = src.shape[1]
        res_id = torch.arange(L).to(src.device).unsqueeze(0)
        device = res_id.device
        bin_values = torch.arange(-8, 9, device=device)
        d = res_id[:, :, None] - res_id[:, None, :]
        bdy = torch.tensor(8, device=device)
        d = torch.minimum(torch.maximum(-bdy, d), bdy)
        d_onehot = (d[..., None] == bin_values).float()
        assert d_onehot.sum(dim=-1).min() == 1
        p = self.linear(d_onehot)
        return p


# ============================================================================
# Utility Functions and Activation Modules
# ============================================================================

def exists(val) -> bool:
    """Check if a value is not None."""
    return val is not None


def default(val, d):
    """Return val if it exists, otherwise return default value d."""
    return val if exists(val) else d


class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x)).
    
    A smooth, non-monotonic activation function that often performs better
    than ReLU in some architectures.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Mish activation.
        
        Note: Inlining this computation saves ~1 second per epoch on V100 GPU
        compared to using a temporary variable.
        """
        return x * (torch.tanh(F.softplus(x)))


def gem(x: torch.Tensor, p: float = 3, eps: float = 1e-6) -> torch.Tensor:
    """Generalized Mean Pooling (GeM).
    
    Computes (mean(x^p))^(1/p) over the last dimension. Generalizes average pooling
    (p=1) and max pooling (p=inf).
    
    Args:
        x: Input tensor.
        p: Power parameter. Higher values approach max pooling.
        eps: Small constant to avoid numerical issues.
        
    Returns:
        Pooled tensor with last dimension reduced.
    """
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    """Generalized Mean Pooling layer with learnable power parameter.
    
    Args:
        p: Initial value for the learnable power parameter.
        eps: Small constant for numerical stability.
    """
    
    def __init__(self, p: float = 3, eps: float = 1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


# ============================================================================
# Dropout Modules (from AlphaFold2 / OpenFold)
# ============================================================================

# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from functools import partialmethod
from typing import Union, List


class Dropout(nn.Module):
    """Dropout with shared mask along specified dimensions.
    
    Implementation of dropout with the ability to share the dropout mask
    along particular dimension(s). This is useful for applying consistent
    dropout to entire rows or columns of pairwise features, as used in AlphaFold2.

    Args:
        r: Dropout rate (probability of zeroing elements).
        batch_dim: Dimension(s) along which the dropout mask is shared.
            Can be a single int or list of ints.
    
    Note:
        If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout with shared mask.
        
        Args:
            x: Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim.
                
        Returns:
            Tensor with dropout applied, same shape as input.
        """
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        x = x * mask
        return x


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class DropoutColumnwise(Dropout):
    """
    Convenience class for columnwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)

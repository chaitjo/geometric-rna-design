import math
import yaml
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import einsum
from einops import rearrange


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)


class RibonanzaNet(nn.Module):

    def __init__(
        self,
        config_filepath="config.yaml",
        checkpoint_filepath="ribonanzanet.pt",
        device="cpu",
    ):
        """
        This class loads the RibonanzaNet model from a configuration file and a checkpoint file,
        and allows a user to predict the reactivity of a single sequence or a batch of sequences.

        Args:
            config_filepath: str
                The path to the configuration file.
            checkpoint_filepath: str
                The path to the checkpoint file.
            device: str
                The device on which to run the model. Default is "cpu".
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
    def predict(self, sequence):
        """
        Predicts the reactivity of a single sequence or a batch of sequences.

        Args:
            sequence: str or list of str
                The sequence(s) for which to predict reactivity.
        
        Returns:
            preds: torch.Tensor
                The predicted reactivity.
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

    def forward(self, src, src_mask=None, return_aw=False):
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

        # save pairwise features for 2D structure prediction
        self.pairwise_features = pairwise_features
        
        output = self.decoder(src).squeeze(-1) + pairwise_features.mean() * 0

        if return_aw:
            return output, attention_weights
        else:
            return output


def mask_diagonal(matrix, mask_value=0):
    matrix = matrix.copy()
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < 4:
                matrix[i][j] = mask_value
    return matrix


class RibonanzaNetSS(RibonanzaNet):

    def __init__(
        self,
        config_filepath="config.yaml",
        checkpoint_filepath="ribonanzanet_ss.pt",
        device="cpu",
    ):
        """
        This class loads the RibonanzaNet SS model from a configuration file and a checkpoint file,
        and allows a user to predict the 2D structure of a single sequence or a batch of sequences.

        Args:
            config_filepath: str
                The path to the configuration file.
            checkpoint_filepath: str
                The path to the checkpoint file.
            device: str
                The device on which to run the model. Default is "cpu".
        """

        super(RibonanzaNetSS, self).__init__(config_filepath, None, device)
        self.dropout = nn.Dropout(0.0)
        self.ct_predictor = nn.Linear(64,1)

        if checkpoint_filepath is not None:
            print(f"Loading RibonanzaNet SS checkpoint: {checkpoint_filepath}")
            self.load_state_dict(torch.load(checkpoint_filepath, map_location="cpu"))

        self.device = device

        # create dummy arnie config
        with open('arnie_file.txt','w+') as f:
            f.write("linearpartition: . \nTMP: /tmp")    
        os.environ['ARNIEFILE'] = 'arnie_file.txt'

    @torch.no_grad()
    def predict(self, sequence):
        """
        Predicts the 2D structure of a single sequence or a batch of sequences.

        Args:
            sequence: str or list of str
                The sequence(s) for which to predict reactivity.
        
        Returns:
            preds: torch.Tensor
                The predicted secondary structure as tensor.
            hungarian_structures: list of str
                The predicted secondary structure as dot-bracket notation.
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
            
        pairwise_features = self.pairwise_features
        # symmetrize
        pairwise_features = pairwise_features + pairwise_features.permute(0,2,1,3)
        # predict
        preds = self.ct_predictor(
            self.dropout(pairwise_features)
        ).sigmoid().squeeze(-1).cpu().numpy()
        # (B, L, L)

        from arnie.pk_predictors import _hungarian
        test_preds_hungarian=[]
        hungarian_structures=[]
        hungarian_bps=[]
        for i in range(len(preds)):
            s,bp = _hungarian(mask_diagonal(preds[i]), theta=0.5, min_len_helix=1) 
            # best theta based on val is 0.5
            hungarian_bps.append(bp)
            ct_matrix = np.zeros((len(s),len(s)))
            for b in bp:
                ct_matrix[b[0],b[1]] = 1
            ct_matrix = ct_matrix + ct_matrix.T
            test_preds_hungarian.append(ct_matrix)
            hungarian_structures.append(s)
        
        return preds, hungarian_structures


class TriangleAttention(nn.Module):
    def __init__(self, in_dim=128, dim=32, n_heads=4, wise="row"):
        super(TriangleAttention, self).__init__()
        self.n_heads = n_heads
        self.wise = wise
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.linear_for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Sigmoid())
        self.to_out = nn.Linear(n_heads * dim, in_dim)

    def forward(self, z, src_mask):
        """
        How to do masking:
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
    def __init__(self, *, dim, hidden_dim=None, mix="ingoing"):
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

    def forward(self, x, src_mask=None):
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

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        pairwise_dimension,
        use_triangular_attention,
        dropout=0.1,
        k=3,
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

    def forward(self, src, pairwise_features, src_mask=None, return_aw=False):

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
    """Multi-Head Attention module"""

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
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

    def forward(self, q, k, v, mask=None, src_mask=None):

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
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, attn_mask=None):

        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature

        if mask is not None:
            attn = attn + mask  # this is actually the bias

        if attn_mask is not None:
            attn = attn.float().masked_fill(attn_mask == -1, float("-1e-9"))

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)

        return output, attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=200):
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

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Outer_Product_Mean(nn.Module):
    def __init__(self, in_dim=256, dim_msa=32, pairwise_dim=64):
        super(Outer_Product_Mean, self).__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)
        self.proj_down2 = nn.Linear(dim_msa**2, pairwise_dim)

    def forward(self, seq_rep, pair_rep=None):
        seq_rep = self.proj_down1(seq_rep)
        outer_product = torch.einsum("bid,bjc -> bijcd", seq_rep, seq_rep)
        outer_product = rearrange(outer_product, "b i j c d -> b i j (c d)")
        outer_product = self.proj_down2(outer_product)

        if pair_rep is not None:
            outer_product = outer_product + pair_rep

        return outer_product


class RelativePositionalEncoding(nn.Module):

    def __init__(self, dim=64):
        super(RelativePositionalEncoding, self).__init__()
        self.linear = nn.Linear(17, dim)

    def forward(self, src):
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


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


def gem(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
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


############################################################################################################

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
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

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
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
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


############################################################################################################

# import re
# import subprocess as sp
# import random
# import string
# import numpy as np

# import string

# # Create lists for uppercase and lowercase alphabets
# uppercase_letters = [letter for letter in string.ascii_uppercase]
# lowercase_letters = [letter for letter in string.ascii_lowercase]


# def complement_to_(string):
#     base_pairing_dct = {'a': 'u', 'u': 'a', 'g': 'c', 'c': 'g', 't': 'a'}
#     return ''.join(base_pairing_dct[x.lower()] for x in string[::-1])


# def run_RNAPVmin(probing_signal, seq, LOC, DEBUG, tauSigmaRatio=1, shapeConversion='S'):
#     reac_file = write_reactivity_file_vienna(probing_signal, seq)
#     fname = write([seq])
#     RNApvmin_command = ['%s/RNApvmin' % LOC, reac_file, '--shapeConversion=%s' %
#                         shapeConversion, '--tauSigmaRatio=%f' % tauSigmaRatio]

#     with open(fname) as f:
#         if DEBUG:
#             print(fname)
#         if DEBUG:
#             print(' '.join(RNApvmin_command))
#         p = sp.Popen(RNApvmin_command, stdin=f, stdout=sp.PIPE, stderr=sp.PIPE)
#     rnapvmin_stdout, rnapvmin_stderr = p.communicate()

#     shape_file = filename()

#     with open(shape_file, 'wb') as f:
#         f.write(rnapvmin_stdout)

#     if DEBUG:
#         print('stdout')
#         print(rnapvmin_stdout)
#         print('stderr')
#         print(rnapvmin_stderr)

#     if p.returncode:
#         raise Exception('RNApvmin failed: on %s\n%s' % (seq, stderr))

#     return shape_file


# ###############################################################################
# # File writing
# ###############################################################################


# def convert_dbn_to_RNAstructure_input(seq, constraints, filename):
#     assert(len(seq) == len(constraints))

#     bp_list = convert_dotbracket_to_bp_dict(constraints)

#     SS_list, pairs_list = [], []

#     for i, (s, c) in enumerate(list(zip(seq, constraints))):
#         if c == 'x':
#             SS_list.append(i+1)
#         elif c == '.':
#             pass
#         elif c == '(':
#             pairs_list.append([i+1, bp_list[i]+1])
#         elif c == ')':
#             pass
#         else:
#             print('Error reading constraint string', c)

#     with open('%s' % filename, 'w') as out:
#         out.write('DS:\n-1\n')
#         out.write('SS:\n')
#         for x in SS_list:
#             out.write('%d\n' % x)
#         out.write('-1\n')
#         out.write('Mod:\n-1\n')
#         out.write('Pairs:\n')
#         for x, y in pairs_list:
#             out.write('%d %d\n' % (x, y))
#         out.write('-1 -1')
#         out.write('FMN:\n-1\nForbids:\n-1\n')


# def write_vector_to_file(vector, outfile):
#     for x in vector:
#         outfile.write('%.3f\n' % x)
#     return


# def write_matrix_to_file(matrix, outfile):
#     for x in matrix:
#         outfile.write('\t'.join(['%.3f' % y for y in x])+'\n')
#     return

# def write_reactivity_file_RNAstructure(reactivities, fname=None):
#     """ writes reactivities (either SHAPE or DMS) to file format used by RNAstructure

#       ex:
#       1 0.120768
#       2 0.190510
#       3 0.155776

#     Args:
#       reactivities (list): a list of normalized reactivity float data. 
#       Negative numbers can be used to indicate no signal.
#     """

#     if fname is None:
#         fname = '%s.SHAPE' % filename()
#     with open(fname, 'w') as f:
#         i = 1
#         for reactivity in reactivities:
#             if reactivity >= 0:
#                 f.write('%d %f\n' % (i, reactivity))
#             i += 1
#     return fname


# def write_reactivity_file_contrafold(reactivities, sequence, fname=None):
#     '''write reactivity (either SHAPE or DMS) to file format used by CONTRAfold.

#       ex:
#       1 U e1 0.120768
#       2 G e1 0.190510
#       3 U e1 0.155776

#     Args:
#       reactivities (list): a list of normalized reactivity float data. 
#       sequence: RNA sequence
#       Negative numbers can be used to indicate no signal.
#     '''

#     assert len(reactivities) == len(sequence)

#     if fname is None:
#         fname = '%s.bpseq' % filename()

#     with open(fname, 'w') as f:
#         i = 1
#         for char, reactivity in list(zip(sequence, reactivities)):
#             if reactivity > 0:
#                 f.write('%d %s e1 %.6f\n' % (i, char, reactivity))
#             elif reactivity == 0:
#                 f.write('%d %s e1 0.0001\n' % (i, char))
#             else:
#                 f.write('%d %s e1 0.0\n' % (i, char))

#             i += 1
#     return fname


# def local_rand_filename(n=6):
#     """generate random filename

#     Args:
#       n (int): number of characters
#     """
#     rand = ''.join([random.choice(string.ascii_lowercase) for _ in range(n)])
#     return rand


# def get_random_folder(n=6):
#     """ generate randome foldername
#     that does not exist in current folder"""
#     out_folder = ''.join([random.choice(string.ascii_lowercase) for _ in range(n)])
#     while os.path.isdir(out_folder):
#         out_folder = ''.join([random.choice(string.ascii_lowercase) for _ in range(n)])
#     tmpdir = load_package_locations()['TMP']
#     out_folder = f'{tmpdir}/{out_folder}'
#     return out_folder


# def filename(n=6):
#     """generate random filename

#     Args:
#       n (int): number of characters
#     """
#     rand = ''.join([random.choice(string.ascii_lowercase) for _ in range(n)])
#     tmpdir = load_package_locations()['TMP']
#     return '%s/%s' % (tmpdir, rand)


# def write(lines, fname=None):
#     """write lines to file

#     Args:
#       lines (list): line(s) to write to file
#       fname (str): filename to write to
#     """
#     if fname is None:
#         fname = '%s.in' % filename()
#     with open(fname, 'w') as f:
#         for line in lines:
#             f.write('%s\n' % line)
#     return fname

# def convert_dbn_to_contrafold_input(seq, constraints, filename):
#     constraint_list = write_constraint_string(seq, constraints)
#     with open('%s' % filename, 'w') as out:
#         for i in range(len(seq)):
#             out.write('%d\t%s\t%d\n' % (i+1, seq[i], constraint_list[i]))


# ###############################################################################
# # File reading
# ###############################################################################

# def bpseq_to_bp_list(bpseq_file, header_length=1):
#     ''' read a bpseq file into a bp_list
#     assumes first line header_length is title 
#     and starts reading only after that
#     '''
#     bps = open(bpseq_file).readlines()
#     bps = bps[header_length:]
#     bp_list = []
#     for bp in bps:
#         bp = bp.split()[::2]
#         bp = [int(nt) for nt in bp]
#         if bp[1] != 0 and bp[0] < bp[1]:
#             bp_list.append([bp[0] - 1, bp[1] - 1])
#     return bp_list


# def ct_to_bp_list(ct_file, header_length=1):
#     ''' read a ct file into a bp_list
#     assumes first line header_length is title 
#     and starts reading only after that
#     '''
#     bps = open(ct_file).readlines()
#     bps = bps[header_length:]
#     bp_list = []
#     for bp in bps:
#         bp = bp.split()[::4]
#         if len(bp) != 0:
#             bp = [int(nt) for nt in bp]
#             if bp[1] != 0 and bp[0] < bp[1]:
#                 bp_list.append([bp[0] - 1, bp[1] - 1])
#     return bp_list


# def prob_to_bpp(prob_file):
#     ''' read a .prob file and return a bpp
#     '''
#     return np.loadtxt(prob_file)


# ###############################################################################
# # Package handling
# ###############################################################################

# supported_packages = [
#     "contrafold",
#     "eternafold",
#     "nupack",
#     "RNAstructure",
#     "RNAsoft",
#     "vienna"
#     # PK Predictors
#     "e2efold",
#     "hotknots", 
#     "inot", 
#     "knotty", 
#     "pknots",
#     "spotrna",
#     "spotrna2",
# ]

# def print_path_files():
#     package_dct = load_package_locations()
#     for key, v in package_dct.items():
#         print(key, v)


# def package_list():
#     pkg_list = []
#     package_dct = load_package_locations()
#     for key, v in package_dct.items():
#         if key != "TMP" and key.lower() != 'bprna':
#             if not key.startswith('linear'):
#                 if key == 'eternafoldparams' and 'eternafold' not in pkg_list:
#                     pkg_list.append('eternafold')
#                 else:
#                     if v != "None":
#                         pkg_list.append(key)
#     return pkg_list


# def load_package_locations(DEBUG=False):
#     '''Set up  paths to RNA folding packages. Checks environment variables or a user-supplied file. 
#     If using the file, specify this in your ~/.bashrc as $ARNIEFILE'''
#     return_dct = {}
#     package_path = os.path.dirname(arnie.__file__)

#     if DEBUG:
#         print(supported_packages)

#     # Read from environment variables
#     for package in supported_packages:
#         envVar = f"{package.upper()}_PATH"
#         # Nupack installation sets its own environment variables
#         if package == "nupack":
#             envVar = f"{package.upper()}HOME"
#         path = os.environ.get(envVar)
#         if path:
#             return_dct[package] = path
#             if DEBUG:
#                 print(f'{package}: {path}')

#     # Read from arnie file as last resort
#     if os.environ.get("ARNIEFILE"):
#         if DEBUG:
#             print('Reading Arnie file at %s' % os.environ['ARNIEFILE'])
#         with open("%s" % os.environ["ARNIEFILE"], 'r') as f:
#             for line in f.readlines():
#                 if line.strip():
#                     if not line.startswith('#'):
#                         key, string = line.split(':')
#                         string = string.strip()
#                         if key not in return_dct:
#                             return_dct[key] = string

#     # If the dict is empty, arnie couldn't find packages in the environment variables or arnie file
#     if not return_dct:
#         raise EnvironmentError("No prediction packages found in your environment. Check your environment variables or your ARNIEFILE.")

#     # Some of the functions currently assume a TMP directory to save temporary files generated by various prediction packages.
#     # If a TMP path is not provided, default to a folder in the current directory.
#     if 'TMP' not in return_dct:
#         tempPath = './tmp'
#         if not os.path.exists(tempPath):
#             os.mkdir(tempPath)
            
#         return_dct['TMP'] = tempPath

#     return return_dct


# ###############################################################################
# # Structure representation conversion
# ###############################################################################

# def convert_dotbracket_to_bp_list(s, allow_pseudoknots=False):
#     if allow_pseudoknots:
#         lefts = ["(", "[", "{", "<"]+uppercase_letters
#         rights = [")", "]", "}", ">"]+lowercase_letters
#         lower_alphabet = [chr(lower) for lower in range(97, 123)]
#         upper_alphabet = [chr(upper) for upper in range(65, 91)]
#         lefts.extend(lower_alphabet)
#         rights.extend(upper_alphabet)
#     else:
#         lefts = ["("]
#         rights = [")"]

#     char_ignored = [char for char in s if char not in rights + lefts + ["."]]
#     char_ignored = list(set(char_ignored))
#     if char_ignored != []:
#         print("WARNING: characters in structure,", char_ignored, "ignored!")

#     l = []
#     for left, right in zip(lefts, rights):
#         bp1 = []
#         bp2 = []
#         for i, char in enumerate(s):
#             if char == left:
#                 bp1.append(i)
#             elif char == right:
#                 bp2.append(i)

#         for i in list(reversed(bp1)):
#             for j in bp2:
#                 if j > i:
#                     l.append([i, j])

#                     bp2.remove(j)
#                     break
#     l = sorted(l, key=lambda x: x[0])
#     return l


# def convert_dotbracket_to_bp_dict(s, allow_pseudoknots=False):

#     if allow_pseudoknots:
#         lefts = ["(", "[", "{", "<"]
#         rights = [")", "]", "}", ">"]
#         lower_alphabet = [chr(lower) for lower in range(97, 123)]
#         upper_alphabet = [chr(upper) for upper in range(65, 91)]
#         lefts.extend(lower_alphabet)
#         rights.extend(upper_alphabet)
#     else:
#         lefts = ["("]
#         rights = [")"]

#     char_ignored = [char for char in s if char not in rights + lefts + ["."]]
#     char_ignored = list(set(char_ignored))
#     if char_ignored != []:
#         print("WARNING: characters in structuture,", char_ignored, "ignored!")

#     m = {}
#     for left, right in zip(lefts, rights):
#         bp1 = []
#         bp2 = []
#         for i, char in enumerate(s):
#             if char == left:
#                 bp1.append(i)
#             elif char == right:
#                 bp2.append(i)

#         for i in list(reversed(bp1)):
#             for j in bp2:
#                 if j > i:
#                     m[i] = j
#                     m[j] = i

#                     bp2.remove(j)
#                     break

#     return m


# def convert_dotbracket_to_matrix(s, allow_pseudoknots=False):
#     matrix = np.zeros([len(s), len(s)])
#     bp_list = convert_dotbracket_to_bp_dict(
#         s, allow_pseudoknots=allow_pseudoknots)
#     for k, v in bp_list.items():
#         matrix[k, v] = 1
#     return matrix


# def convert_bp_list_to_dotbracket(bp_list, seq_len):
#     db = "." * seq_len
#     # group into bps that are not intertwined and can use same brackets!
#     groups = _group_into_non_conflicting_bp(bp_list)

#     # all bp that are not intertwined get (), but all others are
#     # groups to be nonconflicting and then asigned (), [], {}, <> by group
#     chars_set = [("(", ")"), ("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
#     chars_set += [(u,l) for u,l in zip(uppercase_letters,lowercase_letters)]
#     # print(chars_set)
#     # exit()
#     alphabet = [(chr(lower), chr(upper))
#                 for upper, lower in zip(list(range(65, 91)), list(range(97, 123)))]
#     chars_set.extend(alphabet)

#     if len(groups) > len(chars_set):
#         print("WARNING: PK too complex, not enough brackets to represent it.")

#     for group, chars in zip(groups, chars_set):
#         for bp in group:
#             db = db[:bp[0]] + chars[0] + \
#                 db[bp[0] + 1:bp[1]] + chars[1] + db[bp[1] + 1:]
#     return db


# def get_bpp_from_dbn(dbn_struct):
#     """ Gets a base-pairing matrix from a dot-bracket notation structure
#     The matrix has 1's at position i,j if sequence positions i,j are base-paired
#     and 0 otherwise.

#     Args: 
#       dbn_struct - structure in MFE format
#     Return:
#       2D numpy array
#     """

#     bpp_matrix = np.zeros((len(dbn_struct), len(dbn_struct)))

#     bkt_open = "(<{"
#     bkt_close = ")>}"

#     for ii in range(len(bkt_open)):
#         open_char = bkt_open[ii]
#         close_char = bkt_close[ii]

#         open_pos = []

#         for jj in range(len(dbn_struct)):
#             if dbn_struct[jj] == open_char:
#                 open_pos += [jj]
#             if dbn_struct[jj] == close_char:
#                 pair = open_pos[-1]
#                 del open_pos[-1]
#                 bpp_matrix[jj, pair] = 1
#                 bpp_matrix[pair, jj] = 1

#     return bpp_matrix

# def get_helices(s, allowed_buldge_len=0):
#     bp_list = convert_dotbracket_to_bp_list(s, allow_pseudoknots=True)
#     helices = []
#     current_helix = []
#     while bp_list != []:
#         current_bp = bp_list.pop(0)
#         if current_helix == []:
#             current_helix.append(current_bp)
#         else:
#             in_helix_left = list(range(current_helix[-1][0] + 1, current_helix[-1][0] + allowed_buldge_len + 2))
#             in_helix_right = list(range(current_helix[-1][1] - allowed_buldge_len - 1, current_helix[-1][1]))
#             if current_bp[0] in in_helix_left and current_bp[1] in in_helix_right:
#                 current_helix.append(current_bp)
#             else:
#                 helices.append(current_helix)
#                 current_helix = [current_bp]
#     helices.append(current_helix)
#     return helices

# def post_process_struct(s, allowed_buldge_len=0, min_len_helix=1):
#     ''' given a structure, remove any helices that are too short

#     allowed_buldge_len, if 0 does not allow any buldges when calculating the length
#     of the helices, if 1 allows 0-1 and 1-1 buldges, if 2 additionally allows 2-0,2-1,2-2 bulges etc
#     min_len_helix, any helices less than this value will be removed.
#     '''
#     helices = get_helices(s, allowed_buldge_len)
#     bp_list_out = []
#     for helix in helices:
#         if len(helix) >= min_len_helix:
#             bp_list_out.extend(helix)
#     s_out = convert_bp_list_to_dotbracket(bp_list_out, len(s))
#     return s_out


# ###############################################################################
# # Evaluating a structure
# ###############################################################################

# def get_expected_accuracy(dbn_string, bp_matrix, mode='mcc'):
#     '''given a secondary structure as dbn string and base pair matrix, 
#     assess expected accuracy for the structure.

#     Inputs:
#     dbn_string (str): Secondary structure string in dot-parens notation.
#     bp_matrix (NxN array):  symmetric matrix of base pairing probabilities.
#     mode: ['mcc','fscore','sen','ppv']: accuracy metric for which to compute expected value.

#     Returns: expected accuracy value.
#     '''

#     assert bp_matrix.shape[0] == bp_matrix.shape[1]
#     assert bp_matrix.shape[0] == len(dbn_string)

#     struct_matrix = convert_dotbracket_to_matrix(dbn_string)
#     N = len(dbn_string)

#     pred_m = struct_matrix[np.triu_indices(N)]
#     probs = bp_matrix[np.triu_indices(N)]

#     TP = np.sum(np.multiply(pred_m, probs)) + 1e-6
#     TN = 0.5*N*N-1 - np.sum(pred_m) - np.sum(probs) + TP + 1e-6
#     FP = np.sum(np.multiply(pred_m, 1-probs)) + 1e-6
#     FN = np.sum(np.multiply(1-pred_m, probs)) + 1e-6

#     a, b = np.triu_indices(N)
#     cFP = 1e-6  # compatible false positives
#     # for i in range(len(pred_m)):
#     #     if np.sum(struct_matrix,axis=0)[a[i]] + np.sum(struct_matrix,axis=0)[b[i]]==0:
#     #        cFP += np.multiply(pred_m[i], 1-probs[i])

#     if mode == 'sen':
#         return TP/(TP + FN)
#     elif mode == 'ppv':
#         return TP/(TP + FP - cFP)
#     elif mode == 'mcc':
#         return (TP*TN - (FP - cFP)*FN)/np.sqrt((TP + FP - cFP)*(TP + FN)*(TN + FP - cFP)*(TN + FN))
#     elif mode == 'fscore':
#         return 2*TP/(2*TP + FP - cFP + FN)
#     else:
#         print('Error: mode not understood.')


# def get_mean_base_pair_propensity(dbn_string):
#     '''Measure of base pair locality.'''
#     mat = convert_dotbracket_to_matrix(dbn_string)
#     i, j = np.where(mat == 1)
#     # divide by 2 because symmetric matrix
#     mean_bp_dist = 0.5*np.mean([np.abs(x-y) for (x, y) in list(zip(i, j))])
#     return mean_bp_dist

# def is_PK(s):
#     '''return if dotbracket structure represents a PK'''
#     return ("[" in s) or ("{" in s) or ("<" in s)


# def compare_structure_to_native(s, native, metric="all", PK_involved=None):
#     if metric not in ["PPV", "sensitivity", "F1_score", "all"]:
#         raise ValueError(
#             'Only PPV, sensitivity, F1_score and all comparison currently implemented.')

#     if PK_involved is None:
#         s_list = convert_dotbracket_to_bp_list(s, allow_pseudoknots=True)
#         native_list = convert_dotbracket_to_bp_list(
#             native, allow_pseudoknots=True)
#     elif PK_involved:
#         s_list = _seperate_structure_into_PK_involved_or_not(s)["pk_bps"]
#         native_list = _seperate_structure_into_PK_involved_or_not(native)[
#             "pk_bps"]
#     else:
#         s_list = _seperate_structure_into_PK_involved_or_not(s)["no_pk_bps"]
#         native_list = _seperate_structure_into_PK_involved_or_not(native)[
#             "no_pk_bps"]

#     PP = len(s_list)
#     P = len(native_list)

#     TP = len([x for x in s_list if x in native_list])  # true positives
#     FP = len([x for x in s_list if x not in native_list])  # false positives
#     FN = len([x for x in native_list if x not in s_list])  # false negative

#     PPV = TP / PP if PP != 0 else 0
#     sen = TP / P if P != 0 else 0
#     F1 = (2 * PPV * sen) / (sen + PPV) if sen + PPV != 0 else 0

#     if metric == "PPV":
#         return PPV
#     elif metric == "sensitivity":
#         return sen
#     elif metric == "F1_score":
#         return F1
#     elif metric == "all":
#         return {"PPV": PPV, "sensitivity": sen, "F1_score": F1}


# def compare_structures_to_natives(structs, natives, comparison="basepairs", metric="all"):
#     # TODO other metrics? https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values#Negative_predictive_value_(NPV)
#     if comparison not in ["basepairs", "is_PK", "non_PK_basepairs", "PK_basepairs"]:
#         raise ValueError(
#             'Only basepairs and is_PK comparison currently implemented.')
#     if metric not in ["PPV", "sensitivity", "F1_score", "all"]:
#         raise ValueError(
#             'Only PPV, sensitivity, F1_score and all comparison currently implemented.')

#     if comparison == "basepairs":
#         structs_list = [convert_dotbracket_to_bp_list(
#             s, allow_pseudoknots=True) for s in structs]
#         natives_list = [convert_dotbracket_to_bp_list(
#             native, allow_pseudoknots=True) for native in natives]

#         PP = sum(len(x) for x in structs_list)
#         P = sum(len(x) for x in natives_list)

#         TP = sum(len([z for z in x if z in y])
#                  for x, y in zip(structs_list, natives_list))  # true positives

#     elif comparison == "non_PK_basepairs":
#         structs_list = [_seperate_structure_into_PK_involved_or_not(
#             s)["no_pk_bps"] for s in structs]
#         natives_list = [_seperate_structure_into_PK_involved_or_not(
#             native)["no_pk_bps"] for native in natives]
#         PP = sum(len(x) for x in structs_list)
#         P = sum(len(x) for x in natives_list)

#         TP = sum(len([z for z in x if z in y])
#                  for x, y in zip(structs_list, natives_list))  # true positives

#     elif comparison == "PK_basepairs":
#         structs_list = [_seperate_structure_into_PK_involved_or_not(s)[
#             "pk_bps"] for s in structs]
#         natives_list = [_seperate_structure_into_PK_involved_or_not(
#             native)["pk_bps"] for native in natives]
#         PP = sum(len(x) for x in structs_list)
#         P = sum(len(x) for x in natives_list)

#         TP = sum(len([z for z in x if z in y])
#                  for x, y in zip(structs_list, natives_list))  # true positives

#     elif comparison == "is_PK":
#         s_is_PK = [is_PK(s) for s in structs]
#         native_is_PK = [is_PK(native) for native in natives]

#         PP = sum(s_is_PK)
#         P = sum(native_is_PK)

#         # true positives
#         TP = sum([x and y for x, y in zip(s_is_PK, native_is_PK)])
#         # false positives
#         FP = sum([x and not y for x, y in zip(s_is_PK, native_is_PK)])
#         # false negatives
#         FN = sum([not x and y for x, y in zip(s_is_PK, native_is_PK)])
#         TN = sum([not x and not y for x, y in zip(
#             s_is_PK, native_is_PK)])  # true negatives

#     PPV = TP / PP if PP != 0 else 0
#     sen = TP / P if P != 0 else 0
#     F1 = (2 * PPV * sen) / (sen + PPV) if sen + PPV != 0 else 0

#     if metric == "PPV":
#         return PPV
#     elif metric == "sensitivity":
#         return sen
#     elif metric == "F1_score":
#         return F1
#     elif metric == "all":
#         return {"PPV": PPV, "sensitivity": sen, "F1_score": F1}


# ###############################################################################
# # Structure helpers
# ###############################################################################


# def _seperate_structure_into_PK_involved_or_not(s):
#     bp_list = convert_dotbracket_to_bp_list(s, allow_pseudoknots=True)
#     groups = _group_into_non_conflicting_bp(bp_list)
#     bp_list_no_pk = groups[0]
#     bp_list_pk = [bp for group in groups[1:] for bp in group]
#     return {"no_pk_bps": bp_list_no_pk, "pk_bps": bp_list_pk}


# def _get_non_redudant_bp_list(conflict_list):
#     ''' given a conflict list get the list of nonredundant basepairs this list has

#     Args:
#             conflict_list: list of pairs of base_pairs that are intertwined basepairs
#     returns:
#             list of basepairs in conflict list without repeats
#     '''
#     non_redudant_bp_list = []
#     for conflict in conflict_list:
#         if conflict[0] not in non_redudant_bp_list:
#             non_redudant_bp_list.append(conflict[0])
#         if conflict[1] not in non_redudant_bp_list:
#             non_redudant_bp_list.append(conflict[1])
#     return non_redudant_bp_list


# def _get_list_bp_conflicts(bp_list):
#     '''given a bp_list gives the list of conflicts bp-s which indicate PK structure
#     Args:
#             bp_list: of list of base pairs where the base pairs are list of indeces of the bp in increasing order (bp[0]<bp[1])
#     returns:
#             List of conflicting basepairs, where conflicting is pairs of base pairs that are intertwined.
#     '''
#     if len(bp_list) <= 1:
#         return []
#     else:
#         current_bp = bp_list[0]
#         conflicts = []
#         for bp in bp_list[1:]:
#             if (bp[0] < current_bp[1] and current_bp[1] < bp[1]):
#                 conflicts.append([current_bp, bp])
#         return conflicts + _get_list_bp_conflicts(bp_list[1:])


# def _group_into_non_conflicting_bp(bp_list):
#     ''' given a bp_list, group basepairs into groups that do not conflict

#     Args
#             bp_list: list of base_pairs

#     Returns:
#             groups of baspairs that are not intertwined
#     '''
#     conflict_list = _get_list_bp_conflicts(bp_list)

#     non_redudant_bp_list = _get_non_redudant_bp_list(conflict_list)
#     bp_with_no_conflict = [
#         bp for bp in bp_list if bp not in non_redudant_bp_list]
#     groups = [bp_with_no_conflict]
#     while non_redudant_bp_list != []:
#         current_bp = non_redudant_bp_list[0]
#         current_bp_conflicts = []
#         for conflict in conflict_list:
#             if current_bp == conflict[0]:
#                 current_bp_conflicts.append(conflict[1])
#             elif current_bp == conflict[1]:
#                 current_bp_conflicts.append(conflict[0])
#         max_group = [
#             bp for bp in non_redudant_bp_list if bp not in current_bp_conflicts]
#         to_remove = []
#         for i, bpA in enumerate(max_group):
#             for bpB in max_group[i:]:
#                 if bpA not in to_remove and bpB not in to_remove:
#                     if [bpA, bpB] in conflict_list or [bpB, bpA] in conflict_list:
#                         to_remove.append(bpB)
#         group = [bp for bp in max_group if bp not in to_remove]
#         groups.append(group)
#         non_redudant_bp_list = current_bp_conflicts
#         conflict_list = [conflict for conflict in conflict_list if conflict[0]
#                          not in group and conflict[1] not in group]
#     return groups


# ###############################################################################
# # ORPHANED unused anywhere in package
# # figure out utility, take outside of utils and document or depricate
# ###############################################################################


# def convert_multiple_dbns_to_eternafold_input(seq, list_of_constraint_strings, filename):
#     '''hard-coded to have 3 constraints right now for use in eternafold training with kd-ligand data.'''
#     constraint_list = []
#     for constraint_string in list_of_constraint_strings:
#         constraint_list.append(write_constraint_string(seq, constraint_string))

#     with open('%s' % filename, 'w') as out:
#         for i in range(len(seq)):
#             out.write('%d\t%s\t%d\t%d\t%d\n' % (
#                 i+1, seq[i], constraint_list[0][i], constraint_list[1][i], constraint_list[2][i]))


# def write_reactivity_file_vienna(reactivities, sequence, fname=None):
#     '''write reactivity (either SHAPE or DMS) to file format used by ViennaRNA.

#       ex:
#       1 U 0.120768
#       2 G 0.190510
#       3 U 0.155776

#     Args:
#       reactivities (list): a list of normalized reactivity float data. 
#       sequence: RNA sequence
#       Negative numbers can be used to indicate no signal.
#     '''

#     assert len(reactivities) == len(sequence)

#     if fname is None:
#         fname = '%s.SHAPE' % filename()

#     with open(fname, 'w') as f:
#         i = 1
#         for char, reactivity in list(zip(sequence, reactivities)):
#             if reactivity >= 0:
#                 f.write('%d %s %f\n' % (i, char, reactivity))

#             i += 1
#     return fname


# def get_missing_motif_bases(seq):
#     FMN_apt1 = 'AGGAUAU'
#     FMN_apt2 = 'AGAAGG'
#     a = seq.find(FMN_apt1) - 1
#     # print(a)
#     b = seq.find(FMN_apt2) + len(FMN_apt2)
#     #print(seq.find(FMN_apt2), b)

#     return seq[a], seq[b]


# def combo_list_to_dbn_list(seq, final_combo_list, apt_idx_list, apt_ss_list):
#     """ Helper function for write_constraints 
#     Converts the combination of aptamer fragments to dbn strings
#     """
#     dbn_string_list = []
#     for combo in final_combo_list:
#         dbn_string = ['.']*len(seq)
#         temp = apt_idx_list[combo, :][:, 1:3]

#         # Fill in the dbn string with the aptamer ss
#         for (start, finish), apt_ss in zip(temp, apt_ss_list):
#             dbn_string[start:finish] = list(apt_ss)
#         dbn_string = ''.join(dbn_string)

#         # Check if aptamer is flipped
#         if temp[0, 0] > temp[-1, 0]:
#             dbn_string = flip_ss(dbn_string)
#         dbn_string_list.append(dbn_string)
#     return dbn_string_list


# def write_combo_constraints(seq, raw_apt_seq, raw_apt_ss, verbose=False):
#     """ Given a sequence, get all possible secondary constraints of the aptamer 

#     Args:
#       seq: RNA sequence
#       raw_apt_seq: aptamer sequence
#         e.g. CAAAG+CAAAG+GGCCUUUUGGCC
#         + denotes splitable aptamer
#       raw_apt_ss: aptamer secondary structure
#         e.g. (xxx(+)xxx)+((((xxxx))))
#         + denotes splitable aptamer
#         x denotes unpaired base
#         . denotes wildcard (can be anything)
#       verbose: to be verbose
#     Returns
#       list of all possible dbn_string for the given aptamer
#     """
#     if verbose:
#         print('Writing constraints')
#         print(seq)
#         print(raw_apt_seq)
#         print(raw_apt_ss)
#     # split everything by +
#     apt_seq_list = raw_apt_seq.split('+')
#     apt_ss_list = raw_apt_ss.split('+')
#     if len(apt_seq_list) != len(apt_ss_list):
#         raise ValueError(
#             'Missing + in aptamer sequence or secondary structure')

#     # Iterate through each aptamer fragment and save its idx,
#     apt_idx_list = []
#     for idx, (apt_seq, apt_ss) in enumerate(zip(apt_seq_list, apt_ss_list)):
#         if seq.find(apt_seq) == -1:
#             raise ValueError("Aptamer segment {} not found".format(idx+1))
#         if len(apt_seq) != len(apt_ss):
#             raise ValueError(
#                 "Mismatch between aptamer sequence and aptamer secondary structure")

#         # Save all locations of each fragment
#         for m in re.finditer(apt_seq, seq):
#             # Note: cannot get overlapping segments
#             span = m.span()
#             start = span[0]
#             finish = span[1]
#             ss_length = len(apt_seq)
#             apt_idx_list.append([idx, start, finish, ss_length])
#     apt_idx_list = np.array(apt_idx_list)

#     # Now combine aptamer fragments into full secondary structure constraints
#     N_frag = len(apt_ss_list)  # Number of fragments to stitch together

#     # Get a list of all possible combination (each combination is a list as well)
#     temp = np.array([np.where(apt_idx_list[:, 0] == idx)[0]
#                      for idx in range(N_frag)])
#     if len(temp) > 1:
#         combo = np.meshgrid(*temp)
#         # reformat the combinations
#         combo_list = np.array(combo).T.reshape(-1, N_frag)
#     else:
#         combo_list = np.array([[x] for x in temp[0]])

#     # Check each combination to make sure it's feasible, if not remove
#     final_combo_list = prune_combo_list(combo_list, apt_idx_list, N_frag)

#     # Convert each combination into a dbn_string
#     dbn_list = combo_list_to_dbn_list(
#         seq, final_combo_list, apt_idx_list, apt_ss_list)

#     # List will be empty if nothing can be in order (only for >= 3 fragments)
#     return dbn_list


# def write_constraint_string(seq, constraint_dbn):
#     '''write set of integers to represent constraints, i.e. for use in bpseq format.'''

#     assert(len(seq) == len(constraint_dbn))

#     bp_list = convert_dotbracket_to_bp_dict(constraint_dbn)

#     constraint_list = []

#     for i, c in enumerate(constraint_dbn):
#         if c == 'x':
#             constraint = 0
#         elif c == '.':
#             constraint = -1  # or -1 if undefined
#         elif c in ['(', ')']:
#             constraint = bp_list[i]+1
#         else:
#             print('Error reading constraint string', c)
#         constraint_list.append(constraint)
#     return constraint_list


# def prune_combo_list(combo_list, apt_idx_list, N_frag):
#     """ Helper function for write_constraints 
#     Prunes all possible combinations of the aptamer fragments
#     """
#     final_idx_list = []
#     for idx, combo in enumerate(combo_list):
#         temp = apt_idx_list[combo, :]
#         start_idx_list = temp[:, 1]
#         ss_len_list = temp[:, 3]
#         if N_frag == 1:  # If <= 2 fragments than always in order
#             final_idx_list.append(idx)
#         elif N_frag == 2:
#             dx = np.diff(start_idx_list)
#             if dx != 0:
#                 final_idx_list.append(idx)
#         else:
#             # first check if aptamer is in the correct order in the sequence
#             dx = np.diff(start_idx_list)
#             if np.all(dx < 0) or np.all(dx > 0):
#                 # second check if aptamer fragments don't overlap
#                 # each element in dx must be >= length of ss
#                 if dx[0] > 0:  # fragments in increasing order
#                     ss_len_list = ss_len_list[:-1]
#                 else:  # fragments in decreasing order
#                     ss_len_list = ss_len_list[1:]

#                 # Check if there is enough bases to fit the aptamer fragments
#                 if np.all(np.abs(dx) >= ss_len_list):
#                     final_idx_list.append(idx)
#     final_combo_list = combo_list[final_idx_list]
#     return np.array(final_combo_list)


# def flip_ss(ss):
#     """ Flips a secondary structure 
#     Only flips the unpaired bases
#     e.g. (.((....))....( ->
#          ).((....))....)
#     """
#     bp_list = []
#     unmatch_list = []
#     ss = list(ss)
#     for idx, c in enumerate(ss):
#         if c == '(':
#             bp_list.append(idx)
#         elif c == ')':
#             if len(bp_list):
#                 bp_list.pop()
#             else:
#                 unmatch_list.append(idx)
#     unmatch_list += bp_list
#     for idx in unmatch_list:
#         if ss[idx] == '(':
#             ss[idx] = ')'
#         else:
#             ss[idx] = '('
#     ss = ''.join(ss)
#     return ss

# def write_constraints(seq, motif=False, MS2=False, LIG=False, lig1=('nAGGAUAU', '(xxxxxx('), lig2=('AGAAGGn', ')xxxxx)')):
#     '''Inputs:
#     seq: RNA sequence
#     motif: tuple (seq, struct) of motif. For example: PUM would be motif=('UGUAUAUA','xxxxxxxx').
#     MS2: bool, whether to include MS2 constraint or not
#     lig1: tuple (seq, struct) for 5' portion of ligand aptamer. Default is FMN.
#     lig2: tuple (seq, struct) for 3' portion of ligand aptamer

#     Outputs:
#     dbn string, () for paired, x for unpaired, . for unspecified
#     '''

#     # when FMN aptamer and MS2 aptamer overlap, MS2 loses out on bp
#     MS2_apt = 'ACAUGAGGAUCACCCAUGU'
#     LIG_apt1 = lig1[0].replace('n', '')
#     LIG_apt2 = lig2[0].replace('n', '')

#     unpaired_list = []
#     bp_list = {}

#     dbn_string = ['.']*len(seq)

#     if motif:
#         if LIG:
#             raise ValueError(
#                 'Sorry, due to some hacky hard-coding, cannot handle both motif and LIG inputs at this time.')
#         else:
#             return write_constraints(seq, LIG=True, lig1=motif, lig2=('', ''))

#     if LIG:
#         if seq.find(LIG_apt1) == -1:
#             raise RuntimeError("ligand 5' aptamer domain not found, %s" % seq)

#         else:
#             if lig1[0].startswith('n'):
#                 start1 = seq.find(LIG_apt1) + len(LIG_apt1) - len(lig1[0])
#                 if start1 < 0:  # hws: changed from 1, maybe wrong?
#                     start1 = seq.find(
#                         LIG_apt1, start1+len(lig1[0])+1) + len(LIG_apt1) - len(lig1[0])
#             else:
#                 start1 = seq.find(LIG_apt1)
#                 if start1 < 0:  # hws: changed from 1, maybe wrong?
#                     start1 = seq.find(LIG_apt1, start1+len(lig1[0])+1)

#             finish1 = start1 + len(lig1[0])

#             if lig2[0].startswith('n'):
#                 start2 = seq.find(LIG_apt2, finish1+1) + \
#                     len(LIG_apt2) - len(lig2[0])
#             else:
#                 start2 = seq.find(LIG_apt2, finish1+1)
#             finish2 = start2 + len(lig2[0])
#             #print('start1, finish1, start2, finish2 FMN', start1, finish1, start2, finish2)
#             dbn_string[start1:finish1] = list(lig1[1])
#             dbn_string[start2:finish2] = list(lig2[1])

#     if MS2:
#         if seq.find(MS2_apt) == -1:
#             raise RuntimeError("MS2 aptamer domain not found: %s" % seq)
#         else:
#             start = seq.find(MS2_apt)
#             finish = start+len(MS2_apt)
#             #print('start, finish MS2', start, finish)

#             if dbn_string[start] != ".":
#                 #print('warning, aptamer overlap')
#                 dbn_string[start+1:finish-1] = list('((((x((xxxx))))))')
#             else:
#                 dbn_string[start:finish] = list('(((((x((xxxx)))))))')

#     return ''.join(dbn_string)


# ###############################################################################


# from scipy.optimize import linear_sum_assignment


# def _hungarian(bpp, exp=1, sigmoid_slope_factor=None, prob_to_0_threshold_prior=0,
#                prob_to_1_threshold_prior=1, theta=0, ln=False, add_p_unpaired=True,
#                allowed_buldge_len=0, min_len_helix=2):

#     bpp_orig = bpp.copy()

#     if add_p_unpaired:
#         p_unpaired = 1 - np.sum(bpp, axis=0)
#         for i, punp in enumerate(p_unpaired):
#             bpp[i, i] = punp

#     # apply prob_to_0 threshold and prob_to_1 threshold
#     bpp = np.where(bpp < prob_to_0_threshold_prior, 0, bpp)
#     bpp = np.where(bpp > prob_to_1_threshold_prior, 1, bpp)

#     # aply exponential. On second thought this is likely not as helpful as sigmoid since
#     # * for 0 < exp < 1 lower probs will increase more than higher ones (seems undesirable)
#     # * for exp > 1 all probs will decrease, which seems undesirable (but at least lower probs decrease more than higher ones)
#     bpp = np.power(bpp, exp)

#     # apply log which follows botlzamann where -ln(P) porportional to Energy
#     if ln:
#         bpp = np.log(bpp)

#     bpp = np.where(np.isneginf(bpp), -1e10, bpp)
#     bpp = np.where(np.isposinf(bpp), 1e10, bpp)

#     # apply sigmoid modified by slope factor
#     if sigmoid_slope_factor is not None and np.any(bpp):
#         bpp = _sigmoid(bpp, slope_factor=sigmoid_slope_factor)

#         # should think about order of above functions and possibly normalize again here

#         # run hungarian algorithm to find base pairs
#     _, row_pairs = linear_sum_assignment(-bpp)
#     bp_list = []
#     for col, row in enumerate(row_pairs):
#         # if bpp_orig[col, row] != bpp[col, row]:
#         #    print(col, row, bpp_orig[col, row], bpp[col, row])
#         if bpp_orig[col, row] > theta and col < row:
#             bp_list.append([col, row])

#     structure = convert_bp_list_to_dotbracket(bp_list, bpp.shape[0])
#     structure = post_process_struct(structure, allowed_buldge_len, min_len_helix)
#     bp_list = convert_dotbracket_to_bp_list(structure, allow_pseudoknots=True)

#     return structure, bp_list
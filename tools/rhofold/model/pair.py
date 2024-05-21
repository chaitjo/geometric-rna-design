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

from typing import Optional

import torch
import torch.nn as nn
import math
from tools.rhofold.model.primitives import Linear, LayerNorm
from tools.rhofold.utils.chunk_utils import chunk_layer


class PairNet(nn.Module):
    def __init__(self,
                 d_model = 64,
                 d_msa = 21,
                 p_drop = 0.,
                 is_pos_emb = True,
                 ):
        super(PairNet, self).__init__()

        self.pair_emb = PairEmbNet(d_model= d_model,
                                   p_drop = p_drop,
                                   d_seq  = d_msa,
                                   is_pos_emb = is_pos_emb)

    def forward(self, msa_tokens, **unused):
        seq_tokens = msa_tokens[:, 0, :]

        B, L = seq_tokens.shape
        idx = torch.cat([torch.arange(L).long().unsqueeze(0) for i in range(B)], dim=0)

        if idx.device != seq_tokens.device:
            idx = idx.to(seq_tokens.device)

        return self.pair_emb(seq_tokens, idx)


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, p_drop=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.drop = nn.Dropout(p_drop)
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))
        self.register_buffer('div_term', div_term)

    def forward(self, x, idx_s):
        B, L, _, K = x.shape
        K_half = K // 2
        pe = torch.zeros_like(x)
        i_batch = -1
        for idx in idx_s:
            i_batch += 1

            if idx.device != self.div_term.device:
                idx = idx.to(self.div_term.device)

            sin_inp = idx.unsqueeze(1) * self.div_term
            emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
            pe[i_batch, :, :, :K_half] = emb.unsqueeze(1)
            pe[i_batch, :, :, K_half:] = emb.unsqueeze(0)

        x = x + torch.autograd.Variable(pe, requires_grad=False)
        return self.drop(x)

class PairEmbNet(nn.Module):
    def __init__(self, d_model=128, d_seq=21, p_drop=0.1,
                 is_pos_emb = True):
        super(PairEmbNet, self).__init__()
        self.d_model = d_model
        self.d_emb = d_model // 2
        self.emb = nn.Embedding(d_seq, self.d_emb)
        self.projection = nn.Linear(d_model, d_model)

        self.is_pos_emb = is_pos_emb
        if self.is_pos_emb:
            self.pos = PositionalEncoding2D(d_model, p_drop=p_drop)

    def forward(self, seq, idx):

        L = seq.shape[1]
        seq = self.emb(seq)
        left  = seq.unsqueeze(2).expand(-1,-1,L,-1)
        right = seq.unsqueeze(1).expand(-1,L,-1,-1)
        pair = torch.cat((left, right), dim=-1)

        pair = self.projection(pair)
        pair = self.pos(pair, idx) if self.is_pos_emb else pair

        return pair


class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z, n):
        """
        Args:
            c_z:
                Pair transition channel dimension
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension
        """
        super(PairTransition, self).__init__()

        self.c_z = c_z
        self.n = n

        self.layer_norm = LayerNorm(self.c_z)
        self.linear_1 = Linear(self.c_z, self.n * self.c_z)
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_z, c_z)

    def _transition(self, z, mask):
        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)
        
        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z) * mask

        return z

    @torch.jit.ignore
    def _chunk(self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"z": z, "mask": mask},
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
        )


    def forward(self, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # [*, N_res, N_res, 1]
        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            z = self._chunk(z, mask, chunk_size)
        else:
            z = self._transition(z=z, mask=mask)

        return z

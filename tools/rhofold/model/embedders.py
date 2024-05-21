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
from typing import Tuple
from argparse import Namespace

from tools.rhofold.model.primitives import Linear, LayerNorm
from tools.rhofold.utils.tensor_utils import add
import tools.rhofold.model.rna_fm as rna_esm
from tools.rhofold.model.msa import MSANet
from tools.rhofold.model.pair import PairNet
from tools.rhofold.utils.alphabet import RNAAlphabet

def exists(val):
    return val is not None

rna_fm_args = {
    'arch': 'roberta_large',
    'layers': 12,
    'embed_dim': 640,
    'ffn_embed_dim': 5120,
    'attention_heads': 20,
    'max_positions': 1024,
    'sample_break_mode': 'eos',
    'tokens_per_sample': 1023,
    'mask_prob': 0.15,
    'pad': 1, 'eos': 2, 'unk': 3, 'dropout': 0.1,
    'no_seed_provided': False,
    '_name': 'ESM-1b'
}

def load_esm1b_rna_t12(theme="protein"):
    alphabet = rna_esm.Alphabet.from_architecture('roberta_large', theme=theme)
    model_type = rna_esm.ProteinBertModel
    model = model_type(
        Namespace(**rna_fm_args), alphabet,
    )
    return model, alphabet

def esm1b_rna_t12():
    return load_esm1b_rna_t12(theme="rna")

class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """
    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(RecyclingEmbedder, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = 1e8

        self.linear = Linear(self.no_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        # [*, N, C_m]
        m_update = self.layer_norm_m(m)
        if(inplace_safe):
            m.copy_(m_update)
            m_update = m

        # [*, N, N, C_z]
        z_update = self.layer_norm_z(z)
        if(inplace_safe):
            z.copy_(z_update)
            z_update = z

        # This squared method might become problematic in FP16 mode.
        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )
        squared_bins = bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )

        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)

        # [*, N, N, C_z]
        d = self.linear(d)
        z_update = add(z_update, d, inplace_safe)

        return m_update, z_update

import os
class MSAEmbedder(nn.Module):
    """MSAEmbedder """

    def __init__(self,
                 c_m,
                 c_z,
                 rna_fm=None,
                 ):
        super().__init__()

        self.rna_fm, self.rna_fm_reduction = None, None
        self.mask_rna_fm_tokens = False

        self.alphabet = RNAAlphabet.from_architecture('RNA')

        self.msa_emb = MSANet(d_model = c_m,
                               d_msa = len(self.alphabet),
                               padding_idx = self.alphabet.padding_idx,
                               is_pos_emb = True,
                               )

        self.pair_emb = PairNet(d_model = c_z,
                                 d_msa = len(self.alphabet),
                                 )

        self.rna_fm, self.rna_fm_reduction = None, None

        if exists(rna_fm) and rna_fm['enable']:
            # Load RNA-FM model
            self.rna_fm_dim = 640
            self.rna_fm, _ = rna_esm.pretrained.esm1b_rna_t12()
            self.rna_fm.eval()
            for param in self.rna_fm.parameters():
                param.detach_()
            self.rna_fm_reduction = nn.Linear(self.rna_fm_dim + c_m, c_m)

            # rna_fm_ckpt = './rna_fm.pt'
            # if not os.path.exists(rna_fm_ckpt):
            #     torch.save({'model': self.rna_fm.state_dict()}, rna_fm_ckpt)

    def forward(self, tokens, rna_fm_tokens = None, is_BKL = True, **unused):

        assert tokens.ndim == 3
        if not is_BKL:
            tokens = tokens.permute(0, 2, 1)

        B, K, L = tokens.size()# batch_size, num_alignments, seq_len
        msa_fea = self.msa_emb(tokens)

        if exists(self.rna_fm):
            results = self.rna_fm(rna_fm_tokens, need_head_weights=False, repr_layers=[12], return_contacts=False)
            token_representations = results["representations"][12].unsqueeze(1).expand(-1, K, -1, -1)
            msa_fea = self.rna_fm_reduction(torch.cat([token_representations, msa_fea], dim = -1))

        pair_fea = self.pair_emb(tokens, t1ds = None, t2ds = None)

        return msa_fea, pair_fea
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

import copy
import torch
import torch.nn as nn

from tools.rhofold.model.embedders import MSAEmbedder, RecyclingEmbedder
from tools.rhofold.model.e2eformer import E2EformerStack
from tools.rhofold.model.structure_module import StructureModule
from tools.rhofold.model.heads import DistHead, SSHead, pLDDTHead
from tools.rhofold.utils.tensor_utils import add
from tools.rhofold.utils.alphabet import get_features


def exists(val):
    return val is not None

class RhoFold(nn.Module):
    """The rhofold network"""

    def __init__(self, config, device):
        """Constructor function."""

        super().__init__()

        self.config = config
        self.device = device

        self.msa_embedder = MSAEmbedder(
            **config.model.msa_embedder,
        )
        self.e2eformer = E2EformerStack(
            **config.model.e2eformer_stack,
        )
        self.structure_module = StructureModule(
            **config.model.structure_module,
        )
        self.recycle_embnet = RecyclingEmbedder(
            **config.model.recycling_embedder,
        )
        self.dist_head = DistHead(
            **config.model.heads.dist,
        )
        self.ss_head = SSHead(
            **config.model.heads.ss,
        )
        self.plddt_head = pLDDTHead(
            **config.model.heads.plddt,
        )


    def forward_cords(self, tokens, single_fea, pair_fea, seq):

        output = self.structure_module.forward(seq, tokens, { "single": single_fea, "pair": pair_fea } )
        output['plddt'] = self.plddt_head(output['single'][-1])

        return output

    def forward_heads(self, pair_fea):

        output = {}
        output['ss'] = self.ss_head(pair_fea.float())
        output['p'], output['c4_'], output['n'] = self.dist_head(pair_fea.float())

        return output

    def forward_one_cycle(self, tokens, rna_fm_tokens, recycling_inputs, seq):
        '''
        Args:
            tokens: [bs, seq_len, c_z]
            rna_fm_tokens: [bs, seq_len, c_z]
        '''

        device = tokens.device

        msa_tokens_pert = tokens[:, :self.config.globals.msa_depth]

        msa_fea, pair_fea = self.msa_embedder.forward(tokens=msa_tokens_pert,
                                                      rna_fm_tokens=rna_fm_tokens,
                                                      is_BKL=True)

        if exists(self.recycle_embnet) and exists(recycling_inputs):
            msa_fea_up, pair_fea_up = self.recycle_embnet(recycling_inputs['single_fea'],
                                                          recycling_inputs['pair_fea'],
                                                          recycling_inputs["cords_c1'"])
            msa_fea[..., 0, :, :] += msa_fea_up
            pair_fea = add(pair_fea, pair_fea_up, inplace=False)

        msa_fea, pair_fea, single_fea = self.e2eformer(
            m=msa_fea,
            z=pair_fea,
            msa_mask=torch.ones(msa_fea.shape[:3]).to(device),
            pair_mask=torch.ones(pair_fea.shape[:3]).to(device),
            chunk_size=None,
        )

        output = self.forward_cords(tokens, single_fea, pair_fea, seq)

        output.update(self.forward_heads(pair_fea))

        recycling_outputs = {
            'single_fea': msa_fea[..., 0, :, :].detach(),
            'pair_fea': pair_fea.detach(),
            "cords_c1'": output["cords_c1'"][-1].detach(),
        }

        return output, recycling_outputs

    def forward(self,
                tokens,
                rna_fm_tokens,
                seq,
                **kwargs):

        """Perform the forward pass.

        Args:

        Returns:
        """

        recycling_inputs = None

        outputs = []
        for _r in range(self.config.model.recycling_embedder.recycles):
            output, recycling_inputs = \
                self.forward_one_cycle(tokens, rna_fm_tokens, recycling_inputs, seq)
            outputs.append(output)

        return outputs
    
    @torch.no_grad()
    def predict(self, fasta_filepath, output_filepath, use_relax=False, relax_steps=1000):
        """
        Predicts the 3D structure of a single sequence.

        Args:
            fasta_filepath (str): Path to the fasta file containing the sequence.
            output_filepath (str): Path to save the predicted 3D structure to a PDB file.
            use_relax (bool): Whether to perform Amber relaxation (default: False).
            relax_steps (int): Number of Amber relaxation steps to perform (default: 1000).
        """
        data_dict = get_features(fasta_filepath, fasta_filepath)
        outputs = self.forward(
            tokens = data_dict['tokens'].to(self.device), 
            rna_fm_tokens = data_dict['rna_fm_tokens'].to(self.device), 
            seq = data_dict['seq']
        )
        output = outputs[-1]
        node_cords_pred = output['cord_tns_pred'][-1].squeeze(0)

        if use_relax:
            # Change naming if using Amber relaxation
            output_filepath_relaxed = copy.copy(output_filepath)
            output_filepath = f'{output_filepath[:-4]}_unrelaxed.pdb'
        
        # Save designed 3D structure to PDB file
        self.structure_module.converter.export_pdb_file(
            data_dict['seq'],
            node_cords_pred.data.cpu().numpy(),
            path=output_filepath, 
            chain_id=None,
            confidence=output['plddt'][0].data.cpu().numpy(),
            logger=None
        )

        if use_relax:
            # Optional Amber relaxation
            # requires OpenMM: `mamba install openmm=7.7 -c conda-forge`
            from tools.rhofold.relax.relax import AmberRelaxation
            amber_relax = AmberRelaxation(max_iterations=relax_steps)
            amber_relax.process(output_filepath, output_filepath_relaxed)

        return node_cords_pred

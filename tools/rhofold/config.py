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

import ml_collections as mlc

rhofold_config = mlc.ConfigDict(
    {
        "globals": {
            "c_z": 128,
            "c_m": 256,
            "c_t": 64,
            "c_e": 64,
            "c_s": 384,
            'msa_depth': 128,
            'frame_version': 'v5.0',
            "eps": 1e-8,
        },
        "model": {
            "input_embedder": {
                "tf_dim": 22,
                "msa_dim": 49,
                "c_z": 128,
                "c_m": 256,
                "relpos_k": 32,
            },
            'msa_embedder':{
                "c_z": 128,
                "c_m": 256,
                'rna_fm':{
                    'enable': True,
                },
            },
            "recycling_embedder": {
                'recycles': 10,
                "c_z": 128,
                "c_m": 256,
                "min_bin": 2,
                "max_bin": 40,
                "no_bins": 40,
            },
            "e2eformer_stack": {
                "blocks_per_ckpt": 1,
                "c_m": 256,
                "c_z": 128,
                "c_hidden_msa_att": 32,
                "c_hidden_opm": 32,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "c_s": 384,
                "no_heads_msa": 8,
                "no_heads_pair": 4,
                "no_blocks": 12,
                "transition_n": 4,
            },
            "structure_module": {
                "c_s": 384,
                "c_z": 128,
                "c_ipa": 16,
                "c_resnet": 128,
                "no_heads_ipa": 12,
                "no_qk_points": 4,
                "no_v_points": 8,
                "no_blocks": 8,
                "no_transition_layers": 1,
                "no_resnet_blocks": 2,
                "no_angles": 6,
                "trans_scale_factor": 10,
                'refinenet':{
                    'enable': True,
                    'dim': 64,
                    'is_pos_emb': True,
                    'n_layer': 4,
                }
            },
            "heads": {
                "plddt": {
                    "c_in": 384,
                    "no_bins": 50,
                },
                "dist": {
                    "c_in": 128,
                    "no_bins": 40,
                },
                "ss": {
                    "c_in": 128,
                    "no_bins": 1,
                },
            },
        },
    }
)




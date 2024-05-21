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

import os
import torch
import numpy as np
import logging
from collections import defaultdict
from tools.rhofold.utils.constants import RNA_CONSTANTS
from tools.rhofold.utils.rigid_utils import Rigid, calc_rot_tsl, calc_angl_rot_tsl, merge_rot_tsl

class RNAConverter():
    """RNA Structure Converter."""

    def __init__(self):
        """"""

        self.eps = 1e-4
        self.__init()

    def __init(self):
        """"""

        self.cord_dict = defaultdict(dict)
        for resd_name in RNA_CONSTANTS.RESD_NAMES:
            for atom_name, _, cord_vals in RNA_CONSTANTS.ATOM_INFOS_PER_RESD[resd_name]:
                self.cord_dict[resd_name][atom_name] = torch.tensor(cord_vals, dtype=torch.float32)

        trans_dict_all = {}
        for resd_name in RNA_CONSTANTS.RESD_NAMES:
            trans_dict = {}
            cord_dict = {}

            atom_infos = RNA_CONSTANTS.ATOM_INFOS_PER_RESD[resd_name]
            angl_infos = RNA_CONSTANTS.ANGL_INFOS_PER_RESD[resd_name]
            n_angls = len(angl_infos)
            
            for atom_name, idx_rgrp, _ in atom_infos:
                if idx_rgrp == 0:
                    cord_dict[atom_name] = self.cord_dict[resd_name][atom_name]

            trans_dict['omega-main'] = (torch.eye(3, dtype=torch.float32), torch.zeros((3), dtype=torch.float32))
            trans_dict['phi-main'] = (torch.eye(3, dtype=torch.float32), torch.zeros((3), dtype=torch.float32))

            for idx_angl, (angl_name, _, atom_names_sel) in enumerate(angl_infos):
                x1 = cord_dict[atom_names_sel[0]]
                x2 = cord_dict[atom_names_sel[1]]
                x3 = cord_dict[atom_names_sel[2]]
                rot, tsl_vec = calc_rot_tsl(x1, x3, x3 + (x3 - x2))
                trans_dict['%s-main' % angl_name] = (rot, tsl_vec)

                for atom_name, idx_rgrp, _ in atom_infos:
                    if idx_rgrp == idx_angl + 3:
                        cord_dict[atom_name] = tsl_vec + torch.sum(
                            rot * self.cord_dict[resd_name][atom_name].view(1, 3), dim=1)

            for idx_angl_src in range(1, n_angls - 1):
                idx_angl_dst = idx_angl_src + 1
                angl_name_src = angl_infos[idx_angl_src][0]
                angl_name_dst = angl_infos[idx_angl_dst][0]
                rot_src, tsl_vec_src = trans_dict['%s-main' % angl_name_src]
                rot_dst, tsl_vec_dst = trans_dict['%s-main' % angl_name_dst]
                rot = torch.matmul(rot_src.transpose(1, 0), rot_dst)
                tsl_vec = torch.matmul(rot_src.transpose(1, 0), tsl_vec_dst - tsl_vec_src)
                trans_dict['%s-%s' % (angl_name_dst, angl_name_src)] = (rot, tsl_vec)

            trans_dict_all[resd_name] = trans_dict

        self.trans_dict_init = trans_dict_all

    def build_cords(self, seq, fram, angl, rtn_cmsk=False):

        # initialization
        n_resds = len(seq)
        device = angl.device

        angl = angl.squeeze(dim=0) / (torch.norm(angl.squeeze(dim=0), dim=2, keepdim=True) + self.eps)
        rigid = Rigid.from_tensor_7(fram, normalize_quats=True)
        fram = rigid.to_tensor_4x4()
        rot = fram[:,:,:3,:3]
        tsl = fram[:,:,:3,3:].permute(0,1,3,2)

        fram = torch.cat([rot, tsl], dim=2)[:,:,:4,:3].permute(1,0,2,3)
        fmsk = torch.ones((n_resds, 1), dtype=torch.int8, device=device)
        amsk = torch.ones((n_resds, RNA_CONSTANTS.N_ANGLS_PER_RESD_MAX), dtype=torch.int8, device=device)
        cord = torch.zeros((n_resds, RNA_CONSTANTS.ATOM_NUM_MAX, 3), dtype=torch.float32, device=device)
        cmsk = torch.zeros((n_resds, RNA_CONSTANTS.ATOM_NUM_MAX), dtype=torch.int8, device=device)

        for resd_name in RNA_CONSTANTS.RESD_NAMES:
            idxs = [x for x in range(n_resds) if seq[x] == resd_name]
            if len(idxs) == 0:
                continue
            cord[idxs], cmsk[idxs] =\
                self.__build_cord(resd_name, fram[idxs], fmsk[idxs], angl[idxs], amsk[idxs])

        return (cord, cmsk) if rtn_cmsk else (cord)

    def __build_cord(self, resd_name, fram, fmsk, angl, amsk):
        """"""

        # initialization
        device = fram.device
        n_resds = fram.shape[0]
        atom_names_all = RNA_CONSTANTS.ATOM_NAMES_PER_RESD[resd_name]
        atom_names_pad = atom_names_all + ['X'] * (RNA_CONSTANTS.ATOM_NUM_MAX - len(atom_names_all))
        atom_infos_all = RNA_CONSTANTS.ATOM_INFOS_PER_RESD[resd_name]

        cord_dict = defaultdict(
            lambda: torch.zeros((n_resds, 3), dtype=torch.float32, device=device))
        cmsk_vec_dict = defaultdict(lambda: torch.zeros((n_resds), dtype=torch.int8, device=device))

        fram_null = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32, device=device)
        fram_dict = defaultdict(lambda: fram_null.unsqueeze(dim=0).repeat(n_resds, 1, 1))
        fmsk_vec_dict = defaultdict(lambda: torch.zeros((n_resds), dtype=torch.int8, device=device))

        trans_dict = {'main': (fram[:, 0, :3], fram[:, 0, 3])}

        rot_curr, tsl_curr = trans_dict['main']
        atom_names_sel = [x[0] for x in atom_infos_all if x[1] == 0]
        for atom_name_sel in atom_names_sel:
            cord_vec = self.cord_dict[resd_name][atom_name_sel].to(device)
            cord_dict[atom_name_sel] = \
                tsl_curr + torch.sum(rot_curr * cord_vec.view(1, 1, 3), dim=2)
            cmsk_vec_dict[atom_name_sel] = fmsk[:, 0]

        # determine 3D coordinates of atoms belonging to side-chain rigid-groups
        angl_infos_all = RNA_CONSTANTS.ANGL_INFOS_PER_RESD[resd_name]
        rgrp_names_all = ['omega', 'phi'] + [x[0] for x in angl_infos_all]

        for idx_rgrp, rgrp_name_curr in enumerate(rgrp_names_all):
            if rgrp_name_curr in ['omega', 'phi', 'angl_0', 'angl_1']:
                rgrp_name_prev = 'main'
            else:
                rgrp_name_prev = 'angl_%d' % (int(rgrp_name_curr[-1]) - 1)

            rot_prev, tsl_prev = trans_dict[rgrp_name_prev]
            rot_base, tsl_vec_base = \
                self.trans_dict_init[resd_name]['%s-%s' % (rgrp_name_curr, rgrp_name_prev)]
            rot_base = rot_base.unsqueeze(dim=0).to(device)
            tsl_base = tsl_vec_base.unsqueeze(dim=0).to(device)
            
            rot_addi, tsl_addi = calc_angl_rot_tsl(angl[:, idx_rgrp])
            rot_curr, tsl_curr = merge_rot_tsl(
                rot_prev, tsl_prev, rot_base, tsl_base, rot_addi, tsl_addi)
            trans_dict[rgrp_name_curr] = (rot_curr, tsl_curr)

            fram_dict[rgrp_name_curr] = \
                torch.cat([rot_curr, tsl_curr.unsqueeze(dim=1)], dim=1)
            fmsk_vec_dict[rgrp_name_curr] = fmsk[:, 0] * amsk[:, idx_rgrp]

            atom_names_sel = [x[0] for x in atom_infos_all if x[1] == idx_rgrp + 1]
            for atom_name_sel in atom_names_sel:
                cord_vec = self.cord_dict[resd_name][atom_name_sel].to(device)

                cord_dict[atom_name_sel] = \
                    tsl_curr + torch.sum(rot_curr * cord_vec.view(1, 1, 3), dim=2)
                cmsk_vec_dict[atom_name_sel] = fmsk_vec_dict[rgrp_name_curr]

        cmsk = torch.stack([cmsk_vec_dict[x] for x in atom_names_pad][:RNA_CONSTANTS.ATOM_NUM_MAX], dim=1)
        cord = torch.stack([cord_dict[x] for x in atom_names_pad][:RNA_CONSTANTS.ATOM_NUM_MAX], dim=1)

        return cord, cmsk

    def export_pdb_file(self, seq, atom_cords, path, atom_masks=None, confidence=None, chain_id=None, logger = None):
        """Export a PDB file."""

        # configurations
        i_code = ' '
        chain_id = '0' if chain_id is None else chain_id
        occupancy = 1.0
        cord_min = -999.0
        cord_max = 999.0
        seq_len = len(seq)

        n_key_atoms = RNA_CONSTANTS.ATOM_NUM_MAX

        # take all the atom coordinates as valid, if not specified
        if atom_masks is None:
            atom_masks = np.ones(atom_cords.shape[:-1], dtype=np.int8)

        # determine the set of atom names (per residue)
        if atom_cords.ndim == 2:
            if atom_cords.shape[0] == seq_len * n_key_atoms:
                atom_cords = np.reshape(atom_cords, [seq_len, n_key_atoms, 3])
                atom_masks = np.reshape(atom_masks, [seq_len, n_key_atoms])
            else:
                raise ValueError('atom coordinates\' shape does not match the sequence length')

        elif atom_cords.ndim == 3:
            assert atom_cords.shape[0] == seq_len
            atom_cords = atom_cords
            atom_masks = atom_masks

        else:
            raise ValueError('atom coordinates must be a 2D or 3D np.ndarray')

        # reset invalid values in atom coordinates
        atom_cords = np.clip(atom_cords, cord_min, cord_max)
        atom_cords[np.isnan(atom_cords)] = 0.0
        atom_cords[np.isinf(atom_cords)] = 0.0

        # export the 3D structure to a PDB file
        os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
        with open(path, 'w') as o_file:
            n_atoms = 0
            for idx_resd, resd_name in enumerate(seq):
                for idx_atom, atom_name in enumerate(RNA_CONSTANTS.ATOM_NAMES_PER_RESD[resd_name]):

                    temp_factor = 0.0 if confidence is None else \
                        float(100 * confidence.reshape([seq_len])[idx_resd - 1])

                    if atom_masks[idx_resd, idx_atom] == 0:
                        continue
                    n_atoms += 1
                    charge = atom_name[0]
                    line_str = ''.join([
                        'ATOM  ',
                        '%5d' % n_atoms,
                        '  ' + atom_name + ' ' * (3 - len(atom_name)),
                        '   %s' % resd_name,
                        ' %s' % chain_id,
                        ' ' * (4 - len(str(idx_resd + 1))),
                        '%s' % str(idx_resd + 1),
                        '%s   ' % i_code,
                        '%8.3f' % atom_cords[idx_resd, idx_atom, 0],
                        '%8.3f' % atom_cords[idx_resd, idx_atom, 1],
                        '%8.3f' % atom_cords[idx_resd, idx_atom, 2],
                        '%6.2f' % occupancy,
                        '%6.2f' % temp_factor,
                        ' ' * 10,
                        '%2s' % charge,
                        '%2s' % ' ',
                    ])
                    assert len(line_str) == 80, 'line length must be exactly 80 characters: ' + line_str
                    o_file.write(line_str + '\n')

        if logger is not None:
            logger.info(f'    Export PDB file to {path}')

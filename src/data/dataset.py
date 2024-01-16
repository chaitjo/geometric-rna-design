import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as data

from src.data.data_utils import *
from src.data.featurizer import *

from src.constants import RNA_NUCLEOTIDES, RNA_ATOMS, DISTANCE_EPS


class RNADesignDataset(data.Dataset):
    """Multi-state RNA Design Dataset
    
    Builds 3-bead coarse grained representation of an RNA backbone: (P, C4', N1 or N9).

    Returned graph is of type `torch_geometric.data.Data` with attributes:
    - seq        sequence converted to int tensor, shape [num_nodes]
    - node_s     node scalar features, shape [num_nodes, num_conf, num_bb_atoms x 5] 
    - node_v     node vector features, shape [num_nodes, num_conf, 2 + (num_bb_atoms - 1), 3]
    - edge_s     edge scalar features, shape [num_edges, num_conf, num_bb_atoms x num_rbf + num_posenc + num_bb_atoms]
    - edge_v     edge vector features, shape [num_edges, num_conf, num_bb_atoms, 3]
    - edge_index edge indices, shape [2, num_edges]
    - mask       node mask, `False` for nodes with missing data

    Args:
        data_list: List of data samples
        split: train/validation/test split; coords are noised during training
        radius: radial cutoff for drawing edges (currently not used)
        top_k: number of edges to draw per node as destination node
        num_rbf: number of radial basis functions
        num_posenc: number of positional encodings per edge
        max_num_conformers: maximum number of conformers sampled per sequence
        noise_scale: standard deviation of gaussian noise added to coordinates
    """
    def __init__(
            self,
            data_list = [],
            split = 'train',
            radius = 4.5,
            top_k = 10,
            num_rbf = 16,
            num_posenc = 16,
            max_num_conformers = 5,
            noise_scale = 0.1,
            pyrimidine_bb_indices = [
                RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("N1") 
            ],
            purine_bb_indices = [
                RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("N9")
            ],
            distance_eps = DISTANCE_EPS,
            device = 'cpu'
        ):
        super().__init__()

        self.pyrimidine_bb_indices = pyrimidine_bb_indices
        self.purine_bb_indices = purine_bb_indices
        self.featurizer = RNAGraphFeaturizer(
            split=split, radius=radius, top_k=top_k, num_rbf=num_rbf,
            num_posenc=num_posenc, max_num_conformers=max_num_conformers,
            noise_scale=noise_scale, distance_eps=distance_eps, device=device
        )

        # Pre-process raw data to prepare self.data_list
        print(f"\n{split.upper()} DATASET")
        print(f"    Pre-processing {len(data_list)} samples")
        self.data_list = []
        for rna in tqdm(data_list):
            coords_list = []
            for coords in rna['coords_list']:
                # Only keep backbone atom coordinates: num_res x num_atoms x 3
                coords = get_backbone_coords(
                    coords, rna['sequence'],
                    self.pyrimidine_bb_indices,
                    self.purine_bb_indices,
                )
                # Do not add structures with missing coordinates for ALL residues
                if not torch.all((coords == FILL_VALUE).sum(axis=(1,2)) > 0):
                    coords_list.append(coords)
            
            if len(coords_list) > 0:
                # Add processed coords_list to self.data_list
                rna['coords_list'] = coords_list
                self.data_list.append(rna)

        print(f"    Finished: {len(self.data_list)} pre-processed samples")

        # Compute number of nodes per sample (used for batching)
        self.node_counts = [len(entry['sequence']) for entry in self.data_list]
    
    def __len__(self): 
        return len(self.data_list)
    
    def __getitem__(self, i): 
        return self.featurizer(self.data_list[i])


class BatchSampler(data.Sampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.
    
    A `torch.utils.data.Sampler` which samples batches according to a 
    maximum number of graph nodes per batch.
    
    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes_batch: maximum number of nodes in any batch
    :param max_nodes_sample: maximum number of nodes in batches with single
        samples, used for samples with length > `max_nodes_batch`
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(
            self, 
            node_counts, 
            max_nodes_batch=3000, 
            max_nodes_sample=5000,
            shuffle=True
        ):
        
        self.node_counts = node_counts
        self.shuffle = shuffle
        self.max_nodes_batch = max_nodes_batch
        self.max_nodes_sample = max_nodes_sample

        # indices of samples with node count <= max_nodes_batch
        max_nodes = min(max_nodes_batch, max_nodes_sample)
        self.idx = [i for i in range(len(node_counts))  
                        if node_counts[i] <= max_nodes]
        # [indices] of samples with max_nodes_sample >= node count > max_nodes_batch
        # appended to list of batches at the end
        self.batches_single = [[i] for i in range(len(node_counts)) 
                        if max_nodes_sample >= node_counts[i] > max_nodes_batch]

        self._form_batches()
        if len(self.batches_single) > 0:
            # append single samples to batches list
            self.batches += self.batches_single
            if self.shuffle: random.shuffle(self.batches)
    
    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes_batch:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)
    
    def __len__(self): 
        if not self.batches: self._form_batches()
        return len(self.batches)
    
    def __iter__(self):
        if not self.batches: self._form_batches()
        for batch in self.batches: yield batch

import os
import numpy as np
from tqdm import tqdm
import torch

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser

from MDAnalysis.analysis import rms


def process_raw(data_path, save_processed=False):
    p = PDBParser(PERMISSIVE=1, QUIET=1)
    seq_to_structs = {}
    seq_to_ids = {}

    print(f"Loading raw .pdb files from {data_path}:")
    filenames = tqdm(os.listdir(os.path.join(data_path, "raw")))
    for filename in filenames:
        structure_id, file_ext = os.path.splitext(filename)
        filenames.set_description(structure_id)
        if file_ext != ".pdb": continue

        # Load sequence and structure
        structure = p.get_structure(structure_id, os.path.join(data_path, "raw", filename))
        seq, coords = get_seq_and_coords(structure)

        # Basic post processing validation
        if len(seq) <= 1: continue  # Do not include single bases as data points

        # Update dictionary
        if seq in seq_to_structs.keys():
            seq_to_structs[seq].append(coords)
            seq_to_ids[seq].append(structure_id)
        else:
            seq_to_structs[seq] = [coords]
            seq_to_ids[seq] = [structure_id]

    data_list = []
    for seq in seq_to_structs.keys():
        data_list.append({
            'seq': seq,
            'coords_list': seq_to_structs[seq],
            'ids': seq_to_ids[seq]
        })
    if save_processed == True:
        torch.save(data_list, os.path.join(data_path, "processed.pt"))
    
    return data_list


def get_seq_and_coords(structure):
    # Return sequence and coarse-grained coordinates array given PDB structure
    seq = ""
    coords = []
    model = structure[0]
    for chain in model:
        for residue in chain:
            if residue.resname in ["A", "G", "C", "U"]:
                if "P" not in residue: continue
                P_coord = residue["P"].coord

                if "C4'" not in residue: continue
                C_coord = residue["C4'"].coord

                # Pyrimidine (C, U): P, C4', and N1
                if residue.resname in ["C", "U"]:
                    if "N1" not in residue: continue
                    N_coord = residue["N1"].coord
                # Purine (A, G): P, C4', and N9
                else:
                    if "N9" not in residue: continue
                    N_coord = residue["N9"].coord

                seq += residue.resname
                coords.append([P_coord, C_coord, N_coord])
    
    assert len(seq) == len(coords)
    return seq, np.array(coords)


def get_avg_rmsds(data_list):
    
    rmsd_per_item = []  # Ordered list of avg. RMSD per item in data_list
    for data in tqdm(data_list):
        seq, coords_list = data["seq"], data["coords_list"]
        
        if len(coords_list) > 1 and len(seq) > 1:
            # Compute pairwise RMSD among all pairs
            rmsds = []
            for i in range(len(coords_list)):
                for j in range(i+1, len(coords_list)):
                    try:
                        rmsds.append(rms.rmsd(
                            # coords_list[i].reshape(-1, 3), coords_list[j].reshape(-1, 3),  # All CG atoms
                            coords_list[i][:, 1], coords_list[j][:, 1],  # C4' only
                            center=True, superposition=True
                        ))
                    except:
                        # Very short sequences, where this fails 
                        # Fall back to RMSE w/out superposition
                        c_i, c_j = coords_list[i][:, 1], coords_list[j][:, 1]
                        c_i = c_i - np.mean(c_i, axis=0)
                        c_j = c_j - np.mean(c_j, axis=0)
                        rmsds.append(np.sqrt(np.mean((c_i - c_j)**2)))
            
            rmsd_per_item.append(np.mean(rmsds))
        else:
            rmsd_per_item.append(0.0)  # Only one structure available
    
    return rmsd_per_item


def get_k_random_entries(arr_list, k):
    """
    :param arr_list: List of np.array entries
    :param k: number of random entries to be selected from arr_list
    """
    n = len(arr_list)
    arr_list = np.array(arr_list)
    if k > n:
        # If k is greater than the length of the list,
        # return all the entries in the list and pad zeros up to k
        zeros_arr = np.zeros_like(arr_list[0])
        entries_list = np.concatenate((arr_list, [zeros_arr] * (k - n)), axis=0)
        mask = [1]*n + [0]*(k - n)
    else:
        # If k is less than or equal to the length of the list, 
        # randomly select k entries
        rand_idx = np.random.choice(n, size=k, replace=False)
        entries_list =  arr_list[rand_idx]
        mask = [1]*k

    return entries_list, mask

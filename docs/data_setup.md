# Downloading and Preparing Data

gRNAde is trained on RNA structures from the PDB at ≤4A resolution downloaded via [RNASolo](https://rnasolo.cs.put.poznan.pl) with date cutoff: 31 October 2023.

All the data is available on HuggingFace: https://huggingface.co/chaitjo/gRNAde_datasets

## Pre-processed Dataset

The processed dataset can be downloaded from HuggingFace: 

```sh
# Ensure you are in the base directory
cd ~/geometric-rna-design/

# Download processed data using HuggingFace CLI (or manually)
hf download chaitjo/gRNAde_datasets RNASolo_31102023_processed.pt --local-dir data/ --repo-type dataset
```

Each unique RNA sequence is available in the following format (most of the metadata is optional for using gRNAde):
```
{
    'sequence'                   # RNA sequence as a string
    'id_list'                    # list of PDB IDs
    'coords_list'                # list of structures, i.e. 3D coordinates of shape ``(length, 27, 3)``
    'sec_struct_list'            # list of secondary structure strings in dotbracket notation
    'sasa_list'                  # list of per-nucleotide SASA values
    'rfam_list'                  # list of RFAM family IDs
    'eq_class_list'              # list of non-redundant equivalence class IDs
    'type_list'                  # list of structure types (RNA-only, RNA-protein complex, etc.)
    'rmsds_list'                 # dictionary of pairwise C4' RMSD values between structures
    'cluster_seqid0.8'           # cluster ID of sequence identity clustering at 80%
    'cluster_structsim0.45'      # cluster ID of structure similarity clustering at 45%
}
```

## Raw Data

To download all the 3D structured from RNASolo with 31 October 2023 cutoff as raw PDB files using HuggingFace:
```sh
# Ensure you are in the base directory
cd ~/geometric-rna-design/

# Download raw data using HuggingFace CLI (or manually)
hf download chaitjo/gRNAde_datasets RNASolo_31102023_raw.tar.gz --local-dir data/ --repo-type dataset
tar -zxvf RNASolo_31102023_raw.tar.gz
```

This data was downloaded from the RNASolo website on 31 October 2023: 3D (PDB) + all molecules + all members + res. ≤4.0 - https://rnasolo.cs.put.poznan.pl/media/files/zipped/bunches/pdb/all_member_pdb_4_0__3_300.zip (currently not working)

To process the raw PDB files into an ML-ready format (same as `RNASolo_31102023_processed.pt`), install the optional dependencies (US-align, CD-HIT) and run the script [`data/process_data.py`](data/process_data.py).

## Splits for Benchmarking (ICLR 2025)

The current codebase houses code for the stable, experimentally validated version of gRNAde (`v1.0.0`) used in "[Generative inverse design of RNA structure and function with gRNAde](https://www.biorxiv.org/content/10.1101/2025.11.29.691298)".
The code used for the [ICLR 2025 Spotlight paper](https://openreview.net/forum?id=lvw3UgeVxS) is available as release `v0.3.2`: https://github.com/chaitjo/geometric-rna-design/releases/tag/v0.3.2

We continue to provided the splits used in our ICLR 2025 experiments in the `data/` directory:
- Single-state split from [Das et al., 2010](https://www.nature.com/articles/nmeth.1433): `data/das_split.pt` (called the Das split for compatibility with older code)
- Multi-state split of structurally flexible RNAs: `data/structsim_split_v2.pt` (Note that we have deprecated an older version of the multi-state split)

The exact PDB IDs used for each of the splits are also available in the `data/split_ids/` directory, in case you are using a different version of RNAsolo after the 31 October 2023 cutoff.

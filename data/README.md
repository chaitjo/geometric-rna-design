# Downloading Data for gRNAde

gRNAde is trained on all RNA structures from the PDB at ≤4A resolution (12K 3D structures from 4.2K unique RNAs) downloaded via [RNASolo](https://rnasolo.cs.put.poznan.pl) with date cutoff: 31 October 2023.
If you would like to train your own models from scratch, download and extract the raw `.pdb` files via the following script into the `data/raw/` directory (or another location indicated by the `DATA_PATH` environment variable in your `.env` file).

> ❗️ Alternatively to the instructions below, you can download a pre-processed [`.pt`](https://drive.google.com/file/d/1gcUUaRxbGZnGMkLdtVwAILWVerVCbu4Y/view?usp=sharing) file and [`.csv`](https://drive.google.com/file/d/1lbdiE1LfWPReo5VnZy0zblvhVl5QhaF4/view?usp=sharing) metadata, and place them into the `data/` directory.

```sh
# Download structures in pdb format
mkdir ~/geometric-rna-design/data/raw
cd ~/geometric-rna-design/data/raw
curl -O https://rnasolo.cs.put.poznan.pl/media/files/zipped/bunches/pdb/all_member_pdb_4_0__3_300.zip
unzip all_member_pdb_4_0__3_300.zip
rm all_member_pdb_4_0__3_300.zip
```
Manual download link: https://rnasolo.cs.put.poznan.pl/archive.
Select the following for creating the download: 3D (PDB) + all molecules + all members + res. ≤4.0

Next, process the raw PDB files into our ML-ready format, which will be saved under `data/processed.pt`.
```sh
# Process raw data into ML-ready format (this may take several hours)
cd ~/geometric-rna-design/
python data/process_data.py
```

Each RNA will be processed into the following format (most of the metadata is optional for simply using gRNAde):
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

We have provided the splits used in our experiments in the `data/` directory:
- Single-state split from [Das et al., 2010](https://www.nature.com/articles/nmeth.1433): `data/das_split.pt` (called the Das split for compatibility with older code)
- Multi-state split of structurally flexible RNAs: `data/structsim_split.pt`

The precise procedure for creating the splits (which can be used to modify and customise them) can be found in the `notebooks/` directory.
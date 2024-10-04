## Downloading and Preparing Data

gRNAde is trained on all RNA structures from the PDB at ≤4A resolution (12K 3D structures from 4.2K unique RNAs) downloaded via [RNASolo](https://rnasolo.cs.put.poznan.pl) with date cutoff: 31 October 2023.
If you would like to train your own models from scratch, download and extract the raw `.pdb` files via the following script into the `data/raw/` directory (or another location indicated by the `DATA_PATH` environment variable in your `.env` file).

🚨 **Note:** Alternatively to the instructions below, you can download a pre-processed [`.pt`](https://drive.google.com/file/d/1gcUUaRxbGZnGMkLdtVwAILWVerVCbu4Y/view?usp=sharing) file and [`.csv`](https://drive.google.com/file/d/1lbdiE1LfWPReo5VnZy0zblvhVl5QhaF4/view?usp=sharing) metadata, and place them into the `data/` directory.

**Method 1: Script**

```sh
# Download structures in PDB format from RNAsolo (31 October 2023 cutoff)
mkdir ~/geometric-rna-design/data/raw
cd ~/geometric-rna-design/data/raw
gdown https://drive.google.com/uc?id=10NidhkkJ-rkbqDwBGA_GaXs9enEBJ7iQ
tar -zxvf RNAsolo_31102023.tar.gz
```
<details>
<summary>Older instuctions for downloading from RNAsolo (not working)</summary>

```sh
curl -O https://rnasolo.cs.put.poznan.pl/media/files/zipped/bunches/pdb/all_member_pdb_4_0__3_300.zip
unzip all_member_pdb_4_0__3_300.zip
rm all_member_pdb_4_0__3_300.zip
```

</details>

> RNAsolo recently stopped hosting downloads for older versions, such as the 31 October 2023 cutoff that we used in our current work, so you can download the exact data we used via our [Google Drive link](https://drive.google.com/file/d/10NidhkkJ-rkbqDwBGA_GaXs9enEBJ7iQ/).

**Method 2: Manual**

Manual download link: https://rnasolo.cs.put.poznan.pl/archive.
Select the following for creating the download: 3D (PDB) + all molecules + all members + res. ≤4.0

Next, process the raw PDB files into our ML-ready format, which will be saved under `data/processed.pt`. 
You need to install the optional dependencies (US-align, CD-HIT) for processing.
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

## Splits for Benchmarking

We have provided the splits used in our experiments in the `data/` directory:
- Single-state split from [Das et al., 2010](https://www.nature.com/articles/nmeth.1433): `data/das_split.pt` (called the Das split for compatibility with older code)
- Multi-state split of structurally flexible RNAs: `data/structsim_split_v2.pt` (Note that we have deprecated an older version of the multi-state split)

The precise procedure for creating the splits (which can be used to modify and customise them) can be found in the `notebooks/` directory. The exact PDB IDs used for each of the splits are also available in the `data/split_ids/` directory, in case you are using a different version of RNAsolo after the 31 October 2023 cutoff.

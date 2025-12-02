# Generative inverse design of RNA structure and function with gRNAde

An improved version of the gRNAde codebase is coming soon! Till then, check out the [new bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.11.29.691298) where we experimentally validated gRNAde in the wet lab for a range of challenging RNA structure and function design problems. 🎉

---

# gRNAde: Geometric Deep Learning for 3D RNA Inverse Design

**gRNAde** is a **g**eometric deep learning pipeline for 3D **RNA** inverse **de**sign, analogous to [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) for protein design. 

🧬 Tutorial notebook to get started: [gRNAde 101](/tutorial/tutorial.ipynb) <a target="_blank" href="https://colab.research.google.com/drive/16rXKgbGXBBsHvS_2V84WbfKsJYf9lO4Q">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

⚙️ Using gRNAde for custom RNA design scenarios: [Design notebook](/notebooks/design.ipynb) <a target="_blank" href="https://colab.research.google.com/drive/1ajcikLbM9v8_mYwWuZAcVP57nek6UBQD">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

✍️ New to 3D RNA modelling? Here's a curated reading + watch list for beginners: [Resources](https://www.chaitjo.com/post/rna-modelling-and-design/)

📄 For more details on the methodology, see the accompanying ICLR paper: ['gRNAde: Geometric Deep Learning for 3D RNA inverse design'](https://arxiv.org/abs/2305.14749)
> Chaitanya K. Joshi, Arian R. Jamasb, Ramon Viñas, Charles Harris, Simon Mathis, Alex Morehead, and Pietro Liò. gRNAde: Geometric Deep Learning for 3D RNA inverse design. *International Conference on Learning Representations 2025.*
>
>[PDF](https://arxiv.org/abs/2305.14749) | [Tweet](https://twitter.com/chaitjo/status/1662118334412800001) | [Slides](https://www.chaitjo.com/publication/joshi-2023-grnade/gRNAde_slides_CASP_RNA_SIG.pdf)

![](/tutorial/fig/grnade_pipeline.png)

gRNAde generates an RNA sequence conditioned on one or more 3D RNA backbone conformations, i.e. both single- and multi-state **fixed-backbone sequence design**.
RNA backbones are featurized as geometric graphs and processed via a multi-state GNN encoder which is equivariant to 3D roto-translation of coordinates as well as conformer order, followed by conformer order-invariant pooling and sequence design.

## Installation

In order to get started, set up a python environment by following the installation instructions below. 
We have tested gRNAde on Linux with Python 3.10.12 and CUDA 11.8 on NVIDIA A100 80GB GPUs and Intel XPUs, as well as on MacOS (CPU).
```sh
# Clone gRNAde repository
cd ~  # change this to your prefered download location
git clone https://github.com/chaitjo/geometric-rna-design.git
cd geometric-rna-design

# Install mamba (a faster conda)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
source ~/.bashrc
# You may also use conda or virtualenv to create your environment

# Create new environment and activate it
mamba create -n rna python=3.10
mamba activate rna
```

Set up your new python environment, starting with PyTorch and PyG:
```sh
# Install Pytorch on Nvidia GPUs (ensure appropriate CUDA version for your hardware)
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Pytorch Geometric (ensure matching torch + CUDA version to PyTorch)
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
```
<details>
<summary>Install Pytorch/PyG on Intel XPUs (specific to Cambridge's Dawn supercomputer)</summary>

```sh
module load default-dawn
source /usr/local/dawn/software/external/intel-oneapi/2024.0/setvars.sh
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install torch_scatter torch_cluster
```

</details>
<br>

Next, install other compulsory dependencies:
```sh
# Install other python libraries
mamba install jupyterlab matplotlib seaborn pandas biopython biotite -c conda-forge
pip install wandb gdown pyyaml ipdb python-dotenv tqdm cpdb-protein torchmetrics einops ml_collections mdanalysis MDAnalysisTests draw_rna arnie

# Install X3DNA for secondary structure determination
cd ~/geometric-rna-design/tools/
tar -xvzf x3dna-v2.4-linux-64bit.tar.gz
./x3dna-v2.4/bin/x3dna_setup
# Follow the instructions to test your installation

# Install EternaFold for secondary structure prediction
cd ~/geometric-rna-design/tools/
git clone --depth=1 https://github.com/eternagame/EternaFold.git && cd EternaFold/src
make
# Notes: 
# - Multithreaded version of EternaFold did not install for me
# - To install on MacOS, start a shell in Rosetta using `arch -x86_64 zsh`

# Download RhoFold checkpoint (~500MB)
cd ~/geometric-rna-design/tools/rhofold/
gdown https://drive.google.com/uc?id=1To2bjbhQLFx1k8hBOW5q1JFq6ut27XEv
```

<details>
<summary>Optionally, you can also set up some extra tools and dependencies.</summary>

```sh
# (Optional) Install CD-HIT for sequence identity clustering
mamba install cd-hit -c bioconda

# (Optional) Install US-align/qTMclust for structural similarity clustering
cd ~/geometric-rna-design/tools/
git clone https://github.com/pylelab/USalign.git && cd USalign/ && git checkout 97325d3aad852f8a4407649f25e697bbaa17e186
g++ -static -O3 -ffast-math -lm -o USalign USalign.cpp
g++ -static -O3 -ffast-math -lm -o qTMclust qTMclust.cpp

# (Optional) Install ViennaRNA, mainly used for plotting in design notebook
cd ~/geometric-rna-design/tools/
tar -zxvf ViennaRNA-2.6.4.tar.gz
cd ViennaRNA-2.6.4
./configure  # ./configure --enable-macosx-installer
make
sudo make install
```

</details>
<br>

Once your python environment is set up, create your `.env` file with the appropriate environment variables; see the .env.example file included in the codebase for reference. 
```sh
cd ~/geometric-rna-design/
touch .env
```

You're now ready to use gRNAde via [the tutorial](/tutorial/tutorial.ipynb).
In order to train your own models from scratch though, you still need to download and process raw RNA structures from RNAsolo ([instructions below](#downloading-data)).


## Directory Structure and Usage

Detailed usage instructions are available in [the tutorial notebook](/tutorial/tutorial.ipynb).

```
.
├── README.md
├── LICENSE
|
├── gRNAde.py                       # gRNAde python module and command line utility
├── main.py                         # Main script for training and evaluating models
|
├── .env.example                    # Example environment file
├── .env                            # Your environment file
|
├── checkpoints                     # Saved model checkpoints
├── configs                         # Configuration files directory
├── data                            # Dataset and data files directory
├── notebooks                       # Directory for Jupyter notebooks
├── tutorial                        # Tutorial with example usage
|
├── tools                           # Directory for external tools
|   ├── draw_rna                    # RNA secondary structure visualization
|   ├── EternaFold                  # RNA sequence to secondary structure prediction tool
|   ├── RhoFold                     # RNA sequence to 3D structure prediction tool
|   ├── ribonanzanet                # RNA sequence to chemical mapping prediction tool
|   └── x3dna-v2.4                  # RNA secondary structure determination from 3D
|
└── src                             # Source code directory
    ├── constants.py                # Constant values for data, paths, etc.
    ├── evaluator.py                # Evaluation loop and metrics
    ├── layers.py                   # PyTorch modules for building Multi-state GNN models
    ├── models.py                   # Multi-state GNN models for gRNAde
    ├── trainer.py                  # Training loop
    |
    └── data                        # Data-related code
        ├── clustering_utils.py     # Methods for clustering by sequence and structural similarity
        ├── data_utils.py           # Methods for loading PDB files and handling coordinates
        ├── dataset.py              # Dataset and batch sampler class
        ├── featurizer.py           # Featurizer class
        └── sec_struct_utils.py     # Methods for secondary structure prediction and determination
```



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

## Citation

```
@inproceedings{joshi2025grnade,
  title={g{RNA}de: Geometric Deep Learning for 3D RNA inverse design},
  author={Joshi, Chaitanya K and Jamasb, Arian R and Vi{\~n}as, Ramon and Harris, Charles and Mathis, Simon V and Morehead, Alex and Anand, Rishabh and Li{\`o}, Pietro},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
}

@incollection{joshi2024grnade,
  title={g{RNA}de: A Geometric Deep Learning pipeline for 3D RNA inverse design},
  author={Joshi, Chaitanya K and Li{\`o}, Pietro},
  booktitle={RNA Design: Methods and Protocols},
  pages={121--135},
  year={2024},
  publisher={Springer}
}
```

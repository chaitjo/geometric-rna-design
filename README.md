# 💣 gRNAde: Geometric Deep Learning for 3D RNA Inverse Design

**gRNAde** is a **g**eometric deep learning pipeline for 3D **RNA** inverse **de**sign, analogous to [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) for protein design. 

🧬 Tutorial notebook to get started: [gRNAde 101](/tutorial/tutorial.ipynb) <a target="_blank" href="https://colab.research.google.com/drive/16rXKgbGXBBsHvS_2V84WbfKsJYf9lO4Q">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

⚙️ Using gRNAde for custom RNA design scenarios: [Design notebook](/notebooks/design.ipynb) <a target="_blank" href="https://colab.research.google.com/drive/1ajcikLbM9v8_mYwWuZAcVP57nek6UBQD">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

✍️ New to 3D RNA modelling? Here's a currated reading + watch list for beginners: [Resources](/tutorial/README.md)

📄 For more details on the methodology, see the accompanying paper: ['Multi-State RNA Design with Geometric Multi-Graph Neural Networks'](https://arxiv.org/abs/2305.14749)
> Chaitanya K. Joshi, Arian R. Jamasb, Ramon Viñas, Charles Harris, Simon Mathis, and Pietro Liò. Multi-State RNA Design with Geometric Multi-Graph Neural Networks. *ICML Computational Biology Workshop, 2023.*
>
>[PDF](https://arxiv.org/abs/2305.14749.abs) | [Tweet](https://twitter.com/chaitjo/status/1662118334412800001) | [Slides](https://www.chaitjo.com/publication/joshi-2023-grnade/gRNAde_slides_CASP_RNA_SIG.pdf)

![](/tutorial/fig/grnade_pipeline.png)

gRNAde generates an RNA sequence conditioned on one or more 3D RNA backbone conformations, i.e. both single- and multi-state **fixed-backbone sequence design**.
RNA backbones are featurized as geometric graphs and processed via a multi-state GNN encoder which is equivariant to 3D roto-translation of coordinates as well as conformer order, followed by conformer order-invariant pooling and sequence design.

## Installation

In order to get started, set up a python environment by following the installation instructions below. 
We have tested gRNAde on Linux with Python 3.10.12 and CUDA 11.8 on an NVIDIA A100 80GB GPU, as well as on MacOS.
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

Next, install the dependencies within your new python environment.
```sh
# Install Pytorch (ensure appropriate CUDA version for your hardware)
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Pytorch Geometric (ensure matching torch + CUDA version)
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install other dependencies
mamba install mdanalysis MDAnalysisTests jupyterlab matplotlib seaborn pandas networkx biopython biotite torchmetrics lovely-tensors -c conda-forge
pip install wandb pyyaml ipdb python-dotenv tqdm lmdb cpdb-protein

# Install EternaFold for secondary structure prediction
cd ~/rna-inverse-folding/tools/
git clone --depth=1 https://github.com/eternagame/EternaFold.git && cd EternaFold/src
make
# Notes: 
# - Multithreaded version of EternaFold did not install for me
# - To install on MacOS, start a shell in Rosetta using `arch -x86_64 zsh`

# (Optional) Install X3DNA for secondary structure determination
cd ~/rna-inverse-folding/tools/
tar -xvzf x3dna-v2.4-linux-64bit.tar.gz
./x3dna-v2.4/bin/x3dna_setup
# Follow the instructions to test your installation

# (Optional) Install draw_rna for secondary structure visualization
cd ~/rna-inverse-folding/tools/
git clone --depth=1 https://github.com/DasLab/draw_rna.git draw_rna_dir && cd draw_rna_dir
python setup.py install

# (Optional) Install CD-HIT for sequence identity clustering
mamba install cd-hit -c bioconda

# (Optional) Install US-align/qTMclust for structural similarity clustering
cd ~/rna-inverse-folding/tools/
git clone https://github.com/pylelab/USalign.git && cd USalign/ && git checkout 97325d3aad852f8a4407649f25e697bbaa17e186
g++ -static -O3 -ffast-math -lm -o USalign USalign.cpp
g++ -static -O3 -ffast-math -lm -o qTMclust qTMclust.cpp
```

Once your python environment is set up, create your `.env` file with the appropriate environment variables; see the .env.example file included in the codebase for reference. 
```sh
cd ~/rna-inverse-folding/
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
├── main.py                         # Main script for training models
|
├── .env.example                    # Example environment file
├── .env                            # Your environment file
|
├── checkpoints                     # Saved model checkpoints
├── configs                         # Configuration files directory
├── data                            # Dataset and data files directory
├── notebooks                       # Directory for Jupyter notebooks
├── scripts                         # Directory for standalone scripts
├── tutorial                        # Tutorial with example usage
|
├── tools                           # Directory for external tools
|   ├── EternaFold                  # RNA sequence to secondary structure prediction
|   └── x3dna-v2.4                  # RNA secondary structure determination from 3D
|
└── src                             # Source code directory
    ├── constants.py                # Constant values for data, paths, etc.
    ├── layers.py                   # PyTorch modules for building Multi-state GNN models
    ├── models.py                   # Multi-state GNN models for gRNAde
    ├── trainer.py                  # Training and evaluation loops
    |
    └── data                        # Data-related code
        ├── clustering_utils.py     # Methods for clustering by sequence and structural similarity
        ├── data_utils.py           # Methods for loading PDB files and handling coordinates
        ├── dataset.py              # Dataset and batch sampler class
        ├── featurizer.py           # Featurizer class
        └── sec_struct_utils.py     # Methods for secondary structure prediction and determination
```



## Downloading Data

gRNAde is trained on all RNA structures from the PDB at ≤4A resolution (12K 3D structures from 4.2K unique RNAs) downloaded via  [RNASolo](https://rnasolo.cs.put.poznan.pl) on 31 October 2023.
If you would like to train your own models from scratch, download and extract the raw `.pdb` files via the following script into the `data/raw/` directory (or another location indicated by the `DATA_PATH` environment variable in your `.env` file).

> Alternatively to the instructions below, you can download a pre-processed [`.pt`](https://drive.google.com/file/d/1gcUUaRxbGZnGMkLdtVwAILWVerVCbu4Y/view?usp=sharing) file and [`.csv`](https://drive.google.com/file/d/1lbdiE1LfWPReo5VnZy0zblvhVl5QhaF4/view?usp=sharing) metadata, and place them into the `data/` directory.

```sh
# Download structures in pdb format
mkdir ~/rna-inverse-folding/data/raw
cd ~/rna-inverse-folding/data/raw
curl -O https://rnasolo.cs.put.poznan.pl/media/files/zipped/bunches/pdb/all_member_pdb_4_0__3_300.zip
unzip all_member_pdb_4_0__3_300.zip
rm all_member_pdb_4_0__3_300.zip
```
Manual download link: https://rnasolo.cs.put.poznan.pl/archive.
Select the following for creating the download: 3D (PDB) + all molecules + all members + res. ≤4.0

Next, process the raw PDB files into our ML-ready format, which will be saved under  `data/processed.pt`.
```sh
# Process raw data into ML-ready format (this may take several hours)
cd ~/rna-inverse-folding/
python scripts/process_data.py
```

Each RNA will be processed into the following format (most of the metadata is optional for simply using gRNAde):
```
{
    'sequence'                   # RNA sequence as a string
    'id_list'                    # list of PDB IDs
    'coords_list'                # list of 3D coordinates of shape ``(length, 27, 3)``
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

## Citation

```
@inproceedings{joshi2023grnade,
  title={Multi-State RNA Design with Geometric Multi-Graph Neural Networks},
  author={Joshi, Chaitanya K. and Jamasb, Arian R. and Viñas, Ramon and Harris, Charles and Mathis, Simon and Liò, Pietro},
  booktitle={ICML 2023 Workshop on Computation Biology},
  year={2023},
}
```

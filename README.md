# 💣 gRNAde: Geometric RNA Design

**gRNAde** is a geometric deep learning pipeline for 3D RNA inverse design, analogous to [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) for protein design. 

gRNAde generates an RNA sequence conditioned on one or more 3D RNA backbone conformations, i.e. both single- and multi-state **fixed-backbone sequence design**.
RNA backbones are featurized as geometric graphs and processed via a multi-state GNN encoder which is equivariant to 3D roto-translation of coordinates as well as conformer order, followed by conformer order-invariant pooling and sequence design.

![](/tutorial/fig/grnade_pipeline.png)

⚙️ Want to use gRNAde for your own RNA designs? Check out the tutorial notebook: [gRNAde 101](/tutorial/tutorial.ipynb)

📄 For more details on the methodology, see the accompanying paper: ['Multi-State RNA Design with Geometric Multi-Graph Neural Networks'](https://arxiv.org/abs/2305.14749)
> Chaitanya K. Joshi, Arian R. Jamasb, Ramon Viñas, Charles Harris, Simon Mathis, and Pietro Liò. Multi-State RNA Design with Geometric Multi-Graph Neural Networks. *ICML Computational Biology Workshop, 2023.*
>
>[PDF](https://arxiv.org/pdf/2305.14749.pdf) | [Tweet](https://twitter.com/chaitjo/status/1662118334412800001)



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

# Create new environment
mamba create -n rna python=3.10
mamba activate rna

# Install Pytorch (ensure appropriate CUDA version for your hardware)
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Pytorch Geometric (ensure matching torch + CUDA version)
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install other dependencies
mamba install mdanalysis MDAnalysisTests jupyterlab matplotlib seaborn pandas networkx biopython biotite torchmetrics lovely-tensors -c conda-forge
pip install wandb pyyaml ipdb python-dotenv tqdm lmdb cpdb-protein

# Install X3DNA for secondary structure determination
cd ~/rna-inverse-folding/tools/
tar -xvzf x3dna-v2.4-linux-64bit.tar.gz
./x3dna-v2.4/bin/x3dna_setup
# Follow the instructions to test your installation

# Install EternaFold for secondary structure prediction
cd ~/rna-inverse-folding/tools/
git clone --depth=1 https://github.com/eternagame/EternaFold.git && cd EternaFold/src
make
# Notes: 
# - Multithreaded version of EternaFold did not install for me
# - To install on MacOS, start a shell in Rosetta using `arch -x86_64 zsh`

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
If you would like to train your own models from scratch, download and extract the raw `.pdb` files via the following script into the `data/raw/` directory.

```sh
# Download structures in pdb format
mkdir ~/rna-inverse-folding/data/raw
cd ~/rna-inverse-folding/data/raw
curl -O https://rnasolo.cs.put.poznan.pl/media/files/zipped/bunches/pdb/all_member_pdb_4_0__3_300.zip
unzip all_member_pdb_4_0__3_300.zip
rm all_member_pdb_4_0__3_300.zip

# Process raw data into ML-ready format (this may take several hours)
cd ~/rna-inverse-folding/
python scripts/process_data.py
```

Manual download link: https://rnasolo.cs.put.poznan.pl/archive.
Select the following for creating the download: 3D (PDB) + all molecules + all members + res. ≤4.0


## Citation

```
@inproceedings{joshi2023multi,
  title={Multi-State RNA Design with Geometric Multi-Graph Neural Networks},
  author={Joshi, Chaitanya K. and Jamasb, Arian R. and Viñas, Ramon and Harris, Charles and Mathis, Simon and Liò, Pietro},
  booktitle={ICML 2023 Workshop on Computation Biology},
  year={2023},
}
```

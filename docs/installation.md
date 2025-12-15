# Installation

To get started, set up a new python environment by following the installation instructions below. 

We have tested gRNAde on Linux systems with Python 3.10.12 and CUDA 11.8/12.1 on NVIDIA A100 80GB GPUs and Intel XPUs, as well as on MacOS (CPU).

## Clone repository and create new python environment

```sh
# Clone gRNAde repository
cd ~/  # or change this to your prefered download location
git clone https://github.com/chaitjo/geometric-rna-design.git
cd geometric-rna-design

# Create new environment and activate it
mamba create -n rna python=3.10
mamba activate rna
# (you can use conda/virtualenv instead of mamba)
```

## Install packages and dependencies

```sh
# Install Pytorch (ensure appropriate CUDA version for your hardware)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install Pytorch Geometric (ensure matching torch + CUDA version to PyTorch)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric

# Install other python libraries
mamba install jupyterlab matplotlib seaborn pandas biopython biotite -c conda-forge
pip install wandb gdown pyyaml ipdb python-dotenv tqdm cpdb-protein torchmetrics einops ml_collections mdanalysis MDAnalysisTests draw_rna arnie
```

## Download model checkpoints for gRNAde and RibonanzaNet

Model checkpoints are available on HuggingFace: https://huggingface.co/chaitjo/gRNAde

```sh
# Ensure you are in the base directory
cd ~/geometric-rna-design

# Install HuggingFace CLI (https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)
curl -LsSf https://hf.co/cli/install.sh | bash
# alternate: pip install -U "huggingface_hub", or brew install huggingface-cli
hf auth login

# Download all checkpoints: gRNAde, RibonanzaNet, RibonanzaNet-SS, and RhoFold (unused)
hf download chaitjo/gRNAde --local-dir checkpoints/
```

The RhoFold checkpoint is optional, and not used by default.

## Create an env file

Once your python environment is set up, create your `.env` file with the appropriate environment variables.
```sh
cd ~/geometric-rna-design/
touch .env
```

An example `.env.example` file is available (copy-paste contents into your `.env` file):
```sh
export PROJECT_PATH='/home/ckj24/geometric-rna-design/'
export DATA_PATH='/home/ckj24/geometric-rna-design/data/'
export WANDB_PROJECT='gRNAde'
export WANDB_ENTITY='chaitjo'
export WANDB_DIR='/home/ckj24/geometric-rna-design/'
```

## Set up optional tools and dependencies

Extra tools and dependencies, used primarily for data preparation.

```sh
# (Optional) Install EternaFold for secondary structure prediction
cd ~/geometric-rna-design/tools/
git clone --depth=1 https://github.com/eternagame/EternaFold.git && cd EternaFold/src
make
# Notes: 
# - Multithreaded version of EternaFold did not install for me
# - To install on MacOS, start a shell in Rosetta using `arch -x86_64 zsh`

# (Optional) Install X3DNA for secondary structure determination
cd ~/geometric-rna-design/tools/
tar -xvzf x3dna-v2.4-linux-64bit.tar.gz
./x3dna-v2.4/bin/x3dna_setup
# Follow the instructions to test your installation

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

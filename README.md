# 💣 gRNAde: Geometric RNA Design

**gRNAde** is a geometric deep learning pipeline for 3D RNA inverse design conditioned on *multiple* backbone conformations. 
gRNAde explicitly accounts for RNA conformational flexibility via a novel **multi-Graph Neural Network** architecture which independently encodes a set of conformers via message passing.

![](fig/grnade_pipeline.png)

Check out the accompanying paper ['Multi-State RNA Design with Geometric Multi-Graph Neural Networks'](https://arxiv.org/abs/2305.14749), which introduces gRNAde.
> Chaitanya K. Joshi, Arian R. Jamasb, Ramon Viñas, Charles Harris, Simon Mathis, and Pietro Liò. Multi-State RNA Design with Geometric Multi-Graph Neural Networks. *arXiv preprint, 2023.*
>
>[PDF](https://arxiv.org/pdf/2305.14749.pdf) | [Thread](https://twitter.com/chaitjo/status/1662118334412800001)

❗️**Note:** gRNAde is under active development; the `main` branch contains the most recent version of the code and models, but the manuscript may not be updated with the latest results. Please check the ['Releases'](https://github.com/chaitjo/geometric-rna-design/releases) tab to reproduce our results.


## Directory Structure and Usage

```
.
├── README.md
|
├── data                    # Data files directory
├── notebooks               # Jupyter notebooks directory
├── configs                 # Configuration files directory
| 
├── main.py                 # Main script for launching experiments
|
└── src
    ├── models.py           # Multi-GNN encoder layers and model
    ├── train.py            # Helper functions for training and evaluation
    ├── data.py             # RNA inverse design dataset
    ├── data_utils.py       # Helper functions for data preparation
    └── featurisation.py    # Input featurisation helpers
```



## Installation

Our experiments used Python 3.8.16 and CUDA 11.3 on NVIDIA Quadro RTX 8000 GPUs.

```sh
# Create new conda environment
conda create --prefix ./env python=3.8
conda activate ./env

# Install PyTorch (Check CUDA version for GPU!)
# Option 1: CPU
# conda install pytorch==1.12.0 -c pytorch
#
# Option 2: GPU, CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Install dependencies
conda install matplotlib pandas networkx
pip install biopython wandb pyyaml ipdb 
conda install jupyterlab -c conda-forge
conda install -c bioconda cd-hit

# Install PyG (Check CPU/GPU/MacOS)
# Option 1: CPU, MacOS
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cpu.html 
# pip install torch-geometric
#
# Option 2: GPU, CUDA 11.3
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-geometric
#
# Option 3: 
# conda install pyg -c pyg  # CPU/GPU, but may not work on MacOS
```



## Downloading Data

We created a machine learning-ready dataset for RNA inverse design using [RNASolo](https://rnasolo.cs.put.poznan.pl) structures at resolution ≤3A. 
Download and extract the raw `.pdb` files via the following script into the `data/raw/` directory.
Running `main.py` for the first time will process the raw data and save the processed samples as a `.pt` file.

```sh
mkdir data/raw
cd data/raw
curl -O https://rnasolo.cs.put.poznan.pl/media/files/zipped/bunches/pdb/all_member_pdb_3_0__3_280.zip
unzip all_member_pdb_3_0__3_280.zip
rm all_member_pdb_3_0__3_280.zip
```

Manual download link: https://rnasolo.cs.put.poznan.pl/archive.
Select the following for creating the download: 3D (PDB) + all molecules + all members + res. ≤3.0



## Citation

```
@article{joshi2023multi,
  title={Multi-State RNA Design with Geometric Multi-Graph Neural Networks},
  author={Joshi, Chaitanya K. and Jamasb, Arian R. and Viñas, Ramon and Harris, Charles and Mathis, Simon and Liò, Pietro},
  journal={arXiv preprint arXiv:2305.14749},
  year={2023},
}
```

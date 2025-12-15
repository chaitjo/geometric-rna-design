# Expert-Level Design of Pseudoknotted RNA Structures

This directory contains the complete pipeline for the **Eterna OpenKnot Benchmark** validation, a community-wide blinded RNA design competition reported in **Figure 2** and **Supplementary Figure 1** of the paper.

## Experimental Workflow

### 1. Design Generation

**`design.ipynb`**: Generate sequences using gRNAde for all OpenKnot targets in Round 3 and 4

- **Design modes**:
  - `2D mode`: Secondary structure only (dot-bracket notation)
  - `3D mode`: Full 3D backbone coordinates from PDB files
- **Generation**: gRNAde samples with temperature [0.1, 1.0] for diversity
- **Screening**: RibonanzaNet predicts SHAPE reactivity and secondary structure
- **Output**: Up to 1 Million designs saved to `designs/` directory

To download all pre-generated designs used for paper submissions:
```sh
# Ensure you are in the base directory
cd ~/geometric-rna-design

# Download all gRNAde designs for OpenKnot puzzles using HuggingFace CLI (or manually)
hf download chaitjo/gRNAde_datasets --include projects/openknot_benchmark/designs/* --local-dir . --repo-type dataset
```

### 2. Submission Preparation

**`submit.ipynb`**: Prepare top designs for Eterna experimental validation

### 3. Results Analysis and Publication Figures

**`publication_figures.ipynb`**: Generate all figures using experimental validation data

Download official competition results:
```sh
# Ensure you are in the base directory
cd ~/geometric-rna-design

# Download experimental OpenKnot Benchmark data (v3.1.0) using HuggingFace CLI (or manually)
hf download chaitjo/gRNAde_datasets projects/openknot_benchmark/OpenKnotBench_data.v3.1.0.csv --local-dir . --repo-type dataset
```

Data source: [OpenKnotAI Design Data Repository](https://github.com/eternagame/OpenKnotAIDesignData/)

## Directory Structure

- **`designs/`**: Raw design outputs with computational scores (~1M designs per puzzle)
- **`structures/`**: PDB files and secondary structures for all 40 puzzles (P01-P20, Q01-Q20)
- **`figures/`**: Publication-ready figures and structure visualizations
- **`metadata_7a.csv`**: Round 3 puzzle specifications (100 nt)
- **`metadata_7b.csv`**: Round 4 puzzle specifications (240 nt)

# Creating RNA Designs with gRNAde

This guide explains how to use the `design.py` script to generate RNA sequences that fold into target 3D structures and secondary structures. The script provides a production-ready command-line interface with comprehensive configuration options, validation checks, and high-throughput screening capabilities.

## Overview

The `design.py` script implements the complete gRNAde design pipeline described in the paper, consisting of three key stages:

1. **Generation**: gRNAde generates a large library of candidate sequences (typically 100,000s to millions) conditioned on your structural constraints
2. **Screening**: RibonanzaNet evaluates each candidate by predicting its secondary structure and chemical reactivity profile
3. **Initial selection**: Designs are filtered based on configurable metrics and thresholds, with top candidates saved for further analysis and experimental validation

The entire pipeline is GPU-accelerated and can generate and screen one million designs in under 12 hours on a single GPU.

## Quick Start

The fastest way to get started is with the default configuration:

```sh
python design.py --config configs/design.yaml
```

This will:
- Load the target structure from the config file
- Generate diverse candidate sequences using gRNAde
- Screen candidates with RibonanzaNet
- Save filtered designs to `outputs/designs_3d_<timestamp>.csv`

The script uses YAML configuration files to specify all design parameters. An example configuration is provided in `configs/design.yaml`.

For more control, you can override specific parameters:

```sh
python design.py --config configs/design.yaml \
    --pdb_filepath path/to/your/structure.pdb \
    --target_sec_struct "(((...)))...[[[...]]]" \
    --output_dir custom_output/
```

For notebook-based workflows with more customization options, see:
- [`projects/openknot_benchmark/design.ipynb`](../projects/openknot_benchmark/design.ipynb) - Interactive design notebook
- [`projects/rna_polymerase_ribozyme/design_gRNAde.ipynb`](../projects/rna_polymerase_ribozyme/design_gRNAde.ipynb) - Functional design campaign example with additional probabilistic constrains on sequences 

## Design Modes

gRNAde supports two design modes, controlled by the `mode` parameter:

**3D Mode** uses full 3D backbone coordinates from the PDB file. This mode captures tertiary interactions, non-canonical base pairs, and pseudoknot geometry that the 2D-only mode can only capture implicitly. 
Requirements:
- `pdb_filepath`: Path to PDB structure file
- `target_sec_struct`: Target secondary structure in extended dot-bracket notation

**2D Mode** uses only secondary structure constraints without explicit 3D conditioning. Forces gRNAde to implicitly consider 3D interactions. Useful when groundtruth 3D structures are unavailable.
Requirements:
- `native_seq`: A reference sequence (used to initialize dummy 3D coordinates)
- `target_sec_struct`: Target secondary structure in extended dot-bracket notation

## Output Format

Designs are saved to a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `fasta_desc` | Full description with all parameters and scores |
| `sequence` | Designed RNA sequence |
| `model` | Model checkpoint used |
| `seed` | Random seed for this design |
| `temperature` | Sampling temperature used |
| `perplexity` | Model perplexity (lower = higher confidence) |
| `openknot_score` | OpenKnot MCC score |
| `sc_score_ribonanzanet` | SHAPE self-consistency MAE |
| `sc_score_ribonanzanet_ss` | 2D self-consistency score |

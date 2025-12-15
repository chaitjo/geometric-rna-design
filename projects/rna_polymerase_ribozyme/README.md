# Generative Design of RNA Polymerase Ribozymes

This directory contains the complete experimental pipeline for designing and validating functional variants of the 5TU catalytic subunit from the triplet-based RNA polymerase ribozyme (TPR), as reported in **Figure 3** and **Figure 4** of the paper.

## Experimental Workflow

### 1. Design Generation

- **`design_gRNAde.ipynb`**: Generate 1 Million designs using gRNAde conditioned on 5TU structure (PDB: 8T2P) and fitness landscape constraints (created using **`create_fitness_constraints.ipynb`**)
- **`design_rational.ipynb`**: Generate 1 Million rational baseline designs using base-pairing heuristics and the same constraints

Both notebooks output sequences with RibonanzaNet computational scores (MAE, MCC) to `designs/` directory.

### 2. Design Filtering and Library Assembly

- **`prepare_final_designs_gRNAde.ipynb`**: Filter gRNAde designs using RibonanzaNet scores, select top candidates
- **`prepare_final_designs_rational.ipynb`**: Filter rational designs (with/without computational screening)
- **`collate_all_designs.ipynb`**: Combine all designs into unified library (~2,000 variants) with PCR primers for synthesis

Final library saved to `final_designs/` as CSV and FASTA, ordered from Twist Bioscience for experimental validation.

### 3. High-Throughput Functional Screening

Raw Illumina sequencing data from the functional assay can be downloaded from HuggingFace:

```sh
# Ensure you are in the base directory
cd ~/geometric-rna-design

# Download raw sequencing data using HuggingFace CLI (or manually)
hf download chaitjo/gRNAde_datasets --include projects/rna_polymerase_ribozyme/raw_sequencing_data/* --local-dir . --repo-type dataset

# Download processed sequencing reads (demultiplexed, merged, trimmed) using HuggingFace CLI (or manually)
hf download chaitjo/gRNAde_datasets --include projects/rna_polymerase_ribozyme/demultiplexed_reads_merged/rctrim/* --local-dir . --repo-type dataset
```

- **`process_raw_sequencing_data.ipynb`**: Process raw reads using fastp (merge paired-end reads) and cutadapt (demultiplex by barcodes), extract full-length 5TU sequences, filter by quality. Outputs FASTA files to `demultiplexed_reads_merged/rctrim/`.

### 4. Fitness Calculation

- **`calculate_fitness.ipynb`**: Calculate variant fitness from sequencing counts across all pre-selection and post-selection conditions. Computes fractional abundance, wild-type normalized enrichment, and fitness scores.

### 5. Publication Figures

- **`publication_figures.ipynb`**: Generate all figures for the paper.

## Directory Structure

- **`designs/`**: Raw design outputs with computational scores
- **`final_designs/`**: Filtered designs for experimental synthesis
- **`raw_sequencing_data/`**: Raw sequencing data (downloadable from HuggingFace)
- **`demultiplexed_reads_merged/`**: Processed sequencing data (downloadable from HuggingFace)
- **`fitness_landscape_constraints/`**: Position-specific design probabilities from prior fitness landscape
- **`gel_23102025/`**: Low-throughput gel validation data
- **`structures/`**: PDB structures and secondary structure files
- **`figures/`**: Publication-ready figures

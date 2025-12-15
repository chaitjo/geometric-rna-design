"""CLI entrypoint for running gRNAde design.

This script provides a simple interface for generating RNA sequence designs
conditioned on target 3D structures and/or secondary structures.

Usage:
    python design.py --config configs/design.yaml
    python design.py --config configs/design.yaml --pdb_filepath path/to/structure.pdb
"""

import dotenv
dotenv.load_dotenv(".env")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import os
import random
from datetime import datetime
import yaml

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

from src.data.featurizer import RNAGraphFeaturizer
from src.models import gRNAde
from src.evaluator import (
    openknot_score_ribonanzanet,
    self_consistency_score_ribonanzanet,
    self_consistency_score_ribonanzanet_sec_struct
)
from src.constants import NUM_TO_LETTER, PROJECT_PATH, FILL_VALUE

from tools.ribonanzanet.network import RibonanzaNet
from tools.ribonanzanet_sec_struct.network import RibonanzaNetSS


##################################
# Helper Functions
##################################

def create_partial_seq_logit_bias(partial_seq, featurizer, device, model_out_dim=4):
    """Create logit bias from partial sequence constraints.
    
    Args:
        partial_seq (str): Partial sequence with '_' for designable positions
                          and 'A', 'G', 'C', 'U' for fixed positions
        featurizer: RNA graph featurizer with letter_to_num mapping
        device: torch device
        model_out_dim (int): Output dimension of model (4 for RNA)
    
    Returns:
        torch.Tensor: Logit bias matrix (seq_len x out_dim)
    """
    # Convert partial sequence to numerical representation
    _partial_seq = []
    for residue in partial_seq:
        if residue in featurizer.letter_to_num.keys():
            # Fixed nucleotide
            _partial_seq.append(featurizer.letter_to_num[residue])
        else:
            # Designable position (use special token beyond vocab)
            _partial_seq.append(len(featurizer.letter_to_num.keys()))
    
    _partial_seq = torch.as_tensor(_partial_seq, device=device, dtype=torch.long)
    
    # Convert to one-hot and create bias matrix
    logit_bias = F.one_hot(_partial_seq, num_classes=model_out_dim + 1).float()
    
    # Create final logit_bias matrix (remove last column for designable positions)
    # Multiply by large value (100.0) to strongly bias towards fixed positions
    logit_bias = logit_bias[:, :-1] * 100.0
    
    return logit_bias


def set_seed(seed=0, device_type="cpu"):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_config_and_structure(config, raw_data):
    """Validate configuration parameters against loaded structure.
    
    Args:
        config (dict): Configuration dictionary from YAML file
        raw_data (dict): Raw structure data from PDB file
    
    Raises:
        ValueError: If any validation check fails
    """
    print(f"\nValidating configuration...")
    
    # Check 1: Secondary structure length matches sequence length
    if len(config['target_sec_struct']) != len(raw_data['sequence']):
        raise ValueError(
            f"Configuration Error: Secondary structure length ({len(config['target_sec_struct'])}) "
            f"does not match PDB sequence length ({len(raw_data['sequence'])}).\n"
            f"  Secondary structure: {config['target_sec_struct']}\n"
            f"  PDB sequence length: {len(raw_data['sequence'])}\n"
            f"  PDB sequence: {raw_data['sequence'][:50]}{'...' if len(raw_data['sequence']) > 50 else ''}"
        )
    print(f"  ✓ Secondary structure length matches sequence length ({len(config['target_sec_struct'])})")
    
    # Check 2: Native sequence length (if provided) matches PDB sequence length
    if config.get('native_seq') and config['native_seq'] != raw_data['sequence']:
        if len(config.get('native_seq', '')) != len(raw_data['sequence']):
            print(f"  ⚠ Warning: native_seq length ({len(config.get('native_seq', ''))}) differs from PDB sequence length ({len(raw_data['sequence'])})")
        else:
            print(f"  ⚠ Warning: native_seq differs from PDB sequence but has same length")
    
    # Check 3: Valid characters in secondary structure brackets
    valid_sec_struct_chars = set('().[]{}><_-')
    invalid_chars = set(config['target_sec_struct']) - valid_sec_struct_chars
    if invalid_chars:
        raise ValueError(
            f"Configuration Error: Invalid characters in secondary structure: {invalid_chars}\n"
            f"  Valid characters: {valid_sec_struct_chars}"
        )
    print(f"  ✓ Secondary structure format is valid")
    
    # Check 4: Validate model dimensions are consistent
    if config['out_dim'] != 4:
        raise ValueError(
            f"Configuration Error: out_dim must be 4 for RNA (A, G, C, U), got {config['out_dim']}"
        )
    print(f"  ✓ Model output dimension is correct ({config['out_dim']})")
    
    # Check 5: Sampling parameters are reasonable
    if config['temperature_min'] <= 0 or config['temperature_max'] <= 0:
        raise ValueError(
            f"Configuration Error: Temperatures must be positive, got min={config['temperature_min']}, max={config['temperature_max']}"
        )
    if config['temperature_min'] > config['temperature_max']:
        raise ValueError(
            f"Configuration Error: temperature_min ({config['temperature_min']}) must be <= temperature_max ({config['temperature_max']})"
        )
    print(f"  ✓ Temperature range is valid ({config['temperature_min']:.2f} - {config['temperature_max']:.2f})")
    
    # Check 6: Filter metric is valid
    valid_metrics = ['openknot_score', 'sc_score_ribonanzanet', 'sc_score_ribonanzanet_ss']
    if config.get('filter_metric') not in valid_metrics:
        raise ValueError(
            f"Configuration Error: Invalid filter_metric '{config.get('filter_metric')}'. "
            f"Must be one of: {valid_metrics}"
        )
    print(f"  ✓ Filter metric is valid ({config['filter_metric']})")
    
    # Check 7: Partial sequence validation (if provided)
    partial_seq = config.get('partial_seq', None)
    if partial_seq:
        # Validate partial sequence length matches PDB sequence
        if len(partial_seq) != len(raw_data['sequence']):
            raise ValueError(
                f"Configuration Error: partial_seq length ({len(partial_seq)}) "
                f"does not match PDB sequence length ({len(raw_data['sequence'])}).\n"
                f"  partial_seq: {partial_seq}\n"
                f"  PDB sequence length: {len(raw_data['sequence'])}\n"
                f"  Use '_' for designable positions."
            )
        
        # Validate partial_seq contains only valid characters
        valid_chars = set('AGCU_')
        invalid_chars = set(partial_seq) - valid_chars
        if invalid_chars:
            raise ValueError(
                f"Configuration Error: Invalid characters in partial_seq: {invalid_chars}\n"
                f"  Valid characters: A, G, C, U, _ (underscore for unfixed positions)"
            )
        
        print(f"  ✓ Partial sequence constraints are valid")
    
    print(f"  ✓ All validation checks passed!\n")


##################################
# Main Design Function
##################################

def main(config, args):
    """Run the gRNAde design pipeline.
    
    Args:
        config (dict): Configuration dictionary from YAML file
        args: Command-line arguments
    """
    # Set random seed
    set_seed(config['seed'])
    
    # Set device (GPU/CPU)
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
    
    # Print ASCII art banner
    print("\n" + "="*80)
    print("""
          ███████████   ██████   █████   █████████       █████         
         ░░███░░░░░███ ░░██████ ░░███   ███░░░░░███     ░░███          
  ███████ ░███    ░███  ░███░███ ░███  ░███    ░███   ███████   ██████ 
 ███░░███ ░██████████   ░███░░███░███  ░███████████  ███░░███  ███░░███
░███ ░███ ░███░░░░░███  ░███ ░░██████  ░███░░░░░███ ░███ ░███ ░███████ 
░███ ░███ ░███    ░███  ░███  ░░█████  ░███    ░███ ░███ ░███ ░███░░░  
░░███████ █████   █████ █████  ░░█████ █████   █████░░████████░░██████ 
 ░░░░░███░░░░░   ░░░░░ ░░░░░    ░░░░░ ░░░░░   ░░░░░  ░░░░░░░░  ░░░░░░  
 ███ ░███                                                              
░░██████                                                               
 ░░░░░░                                                                
    """)
    print("="*80)
    print(f"Device: {device}")
    print(f"Mode: {config['mode']}")
    print(f"Target structure: {args.pdb_filepath or config['pdb_filepath']}")
    
    # Override config with command-line arguments if provided
    if args.pdb_filepath:
        config['pdb_filepath'] = args.pdb_filepath
    if args.target_sec_struct:
        config['target_sec_struct'] = args.target_sec_struct
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Create RNA graph featurizer
    print(f"\nInitializing RNA graph featurizer for {config['mode']} design...")
    featurizer = RNAGraphFeaturizer(
        split="test" if config['mode'] == '3d' else "test_2d",
        radius=config['radius'],
        top_k=config['top_k'],
        num_rbf=config['num_rbf'],
        num_posenc=config['num_posenc'],
        max_num_conformers=config['max_num_conformers'],
        noise_scale=config['noise_scale'],
        drop_prob_3d=config['drop_prob_3d'],
    )
    
    # Initialize gRNAde model
    print(f"Initializing gRNAde model...")
    model = gRNAde(
        node_in_dim=tuple(config['node_in_dim']),
        node_h_dim=tuple(config['node_h_dim']),
        edge_in_dim=tuple(config['edge_in_dim']),
        edge_h_dim=tuple(config['edge_h_dim']),
        num_layers=config['num_layers'],
        drop_rate=config['drop_rate'],
        out_dim=config['out_dim'],
    )
    model_path = os.path.join(PROJECT_PATH, config['model_path'])
    print(f"Loading checkpoint: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    
    # Initialize evaluation models
    print(f"\nInitializing evaluation models...")
    ribonanza_net = RibonanzaNet(
        os.path.join(PROJECT_PATH, 'tools/ribonanzanet/config.yaml'),
        os.path.join(PROJECT_PATH, 'checkpoints/ribonanzanet/ribonanzanet.pt'),
        device
    )
    ribonanza_net = ribonanza_net.to(device)
    ribonanza_net.eval()
    
    ribonanza_net_ss = RibonanzaNetSS(
        os.path.join(PROJECT_PATH, 'tools/ribonanzanet_sec_struct/config.yaml'),
        os.path.join(PROJECT_PATH, 'checkpoints/ribonanzanet_sec_struct/ribonanzanet_ss.pt'),
        device
    )
    ribonanza_net_ss = ribonanza_net_ss.to(device)
    ribonanza_net_ss.eval()
    
    # Load target structure
    print(f"\nLoading target structure...")
    print(f"Target secondary structure: {config['target_sec_struct']}")
    _, raw_data = featurizer.featurize_from_pdb_file(config['pdb_filepath'])
    print(f"Native sequence: {raw_data['sequence']}")
    print(f"Sequence length: {len(raw_data['sequence'])}")
    
    # Run validation checks
    validate_config_and_structure(config, raw_data)
    
    # Process partial sequence constraints (hard constraints)
    partial_seq = config.get('partial_seq', None)
    logit_bias = None
    
    if partial_seq:
        # Count fixed and designable positions
        n_fixed = sum(1 for c in partial_seq if c != '_')
        n_designable = len(partial_seq) - n_fixed
        
        print(f"\nPartial sequence constraints (hard constraints):")
        print(f"  Partial sequence: {partial_seq}")
        print(f"  Fixed positions: {n_fixed} ({100*n_fixed/len(partial_seq):.1f}%)")
        print(f"  Designable positions: {n_designable} ({100*n_designable/len(partial_seq):.1f}%)")
        print(f"\n  Note: For soft constraints (probabilistic design), see:")
        print(f"  projects/rna_polymerase_ribozyme/design_gRNAde.ipynb")
        
        # Create logit bias from partial sequence
        logit_bias = create_partial_seq_logit_bias(
            partial_seq, featurizer, device, model.out_dim
        )
    
    # Create output directory
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"designs_{config['mode']}_{current_datetime}.csv")
    
    print(f"\nOutput file: {output_file}")
    with open(output_file, 'w') as f_out:
        f_out.write("fasta_desc,sequence,model,seed,temperature,perplexity,openknot_score,sc_score_ribonanzanet,sc_score_ribonanzanet_ss\n")
    
    # Prepare data for 2D-only mode
    if config['mode'] == '2d':
        raw_data_2d = {
            "sequence": config.get('native_seq', raw_data['sequence']),
            "coords_list": [torch.ones(len(config['target_sec_struct']), 3, 3) * FILL_VALUE],
            "sec_struct_list": [config['target_sec_struct']],
        }
        featurized_data = featurizer(raw_data_2d).to(device)
    
    # Run design loop
    print(f"\n{'='*60}")
    print(f"Starting design generation...")
    print(f"Total samples to generate: {config['total_samples']:,}")
    print(f"Batch size: {config['n_samples']}")
    print(f"Temperature range: [{config['temperature_min']}, {config['temperature_max']}]")
    print(f"Initial filter metric: {config['filter_metric']}")
    print(f"Initial filter threshold: {config['pass_threshold']}")
    print(f"{'='*60}\n")
    
    designs = []
    try:
        t = tqdm(range(config['total_samples'] // config['n_samples']))
        for _ in t:
            t.set_description(f"Designs collected: {len(designs)}")
            
            # Set temperature and seed for this batch
            temperature = np.random.uniform(config['temperature_min'], config['temperature_max'])
            seed = random.randint(0, 9999)
            set_seed(seed)
            
            # Featurize data
            if config['mode'] == '3d':
                featurized_data = featurizer(raw_data).to(device)
            
            # Sample sequences from model (with optional partial sequence constraints)
            if logit_bias is not None:
                samples, logits = model.sample(
                    featurized_data, config['n_samples'], temperature, 
                    logit_bias=logit_bias, return_logits=True
                )
            else:
                samples, logits = model.sample(
                    featurized_data, config['n_samples'], temperature, return_logits=True
                )
            
            # Calculate perplexity
            n_nodes = logits.shape[1]
            perplexity = (
                torch.exp(
                    F.cross_entropy(
                        logits.view(config['n_samples'] * n_nodes, model.out_dim),
                        samples.view(config['n_samples'] * n_nodes).long(),
                        reduction="none",
                    )
                    .view(config['n_samples'], n_nodes)
                    .mean(dim=1)
                )
                .cpu()
                .numpy()
            )
            
            # Calculate OpenKnot score
            openknot_scores = openknot_score_ribonanzanet(
                samples.cpu().numpy(),
                config['target_sec_struct'],
                featurized_data.mask_seq.cpu().numpy(),
                ribonanza_net
            )
            
            # Calculate SHAPE self-consistency score
            sc_score_ribonanzanet = self_consistency_score_ribonanzanet(
                samples.cpu().numpy(),
                raw_data['sequence'],
                featurized_data.mask_seq.cpu().numpy(),
                ribonanza_net,
            )
            
            # Calculate 2D self-consistency score
            sc_score_ribonanzanet_ss = self_consistency_score_ribonanzanet_sec_struct(
                samples.cpu().numpy(),
                config['target_sec_struct'],
                featurized_data.mask_seq.cpu().numpy(),
                ribonanza_net_ss,
            )
            
            # Filter designs based on configured metric and threshold
            filter_metric = config.get('filter_metric', 'openknot_score')
            if filter_metric == 'openknot_score':
                idx_good_designs = (openknot_scores >= config['pass_threshold'])
            elif filter_metric == 'sc_score_ribonanzanet':
                idx_good_designs = (sc_score_ribonanzanet <= config['pass_threshold'])
            elif filter_metric == 'sc_score_ribonanzanet_ss':
                idx_good_designs = (sc_score_ribonanzanet_ss >= config['pass_threshold'])
            else:
                raise ValueError(f"Unknown filter_metric: {filter_metric}. Choose from: openknot_score, sc_score_ribonanzanet, sc_score_ribonanzanet_ss")
            
            if np.sum(idx_good_designs) > 0:
                samples_filtered = samples.cpu().numpy()[idx_good_designs]
                perplexity_filtered = perplexity[idx_good_designs]
                openknot_scores_filtered = openknot_scores[idx_good_designs]
                sc_score_rnet_filtered = sc_score_ribonanzanet[idx_good_designs]
                sc_score_rnetss_filtered = sc_score_ribonanzanet_ss[idx_good_designs]
                
                # Save designs to file
                with open(output_file, 'a') as f_out:
                    for seq, perp, ok_score, sc_rnet, sc_rnetss in zip(
                        samples_filtered, perplexity_filtered, openknot_scores_filtered, 
                        sc_score_rnet_filtered, sc_score_rnetss_filtered
                    ):
                        seq_str = "".join([NUM_TO_LETTER[num] for num in seq])
                        model_name = os.path.split(model_path)[-1]
                        design = [
                            f"{model_name} seed={seed} temperature={temperature:.4f} perplexity={perp:.4f} openknot_score={ok_score:.4f} sc_score_ribonanzanet={sc_rnet:.4f} sc_score_ribonanzanet_ss={sc_rnetss:.4f}",
                            seq_str, model_name, seed, temperature, perp, ok_score, sc_rnet, sc_rnetss
                        ]
                        designs.append(design)
                        f_out.write(",".join(map(str, design)) + "\n")
            
            # Stop if enough designs collected
            if len(designs) >= config['n_pass']:
                break
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"⚠ Design process interrupted by user (Ctrl+C)")
        print(f"Designs collected before interruption: {len(designs)}")
        print(f"{'='*60}")
    
    finally:
        # Remove duplicates and finalize output
        print(f"\n{'='*60}")
        print(f"Post-processing designs...")
        
        # Check if any designs were generated
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            all_designs_df = pd.read_csv(output_file)
            if len(all_designs_df) > 0:
                print(f"Total designs generated: {len(all_designs_df):,}")
                all_designs_df = all_designs_df.drop_duplicates(subset="sequence")
                print(f"Unique designs: {len(all_designs_df):,}")
                all_designs_df.to_csv(output_file, index=False)
                print(f"\nDesigns saved to: {output_file}")
            else:
                print(f"No designs were generated.")
        else:
            print(f"No designs were generated.")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate RNA sequence designs with gRNAde",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/design.yaml",
        help="Path to design configuration YAML file"
    )
    parser.add_argument(
        "--pdb_filepath",
        type=str,
        default=None,
        help="Path to target PDB structure (overrides config)"
    )
    parser.add_argument(
        "--target_sec_struct",
        type=str,
        default=None,
        help="Target secondary structure in dot-bracket notation (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for designs (overrides config)"
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Print configuration
    print("\nConfiguration:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    
    # Run main function
    main(config, args)

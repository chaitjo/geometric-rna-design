"""CLI entrypoint for training and evaluating gRNAde.

This module wires together configuration, dataset construction, model instantiation,
training, and evaluation. It provides a simple command-line interface backed by wandb
for experiment tracking.

Overview:
    - Loads configuration from a YAML file passed to wandb (`--config`)
    - Initializes the device (CPU/CUDA) and reproducible seeds
    - Builds datasets and dataloaders with memory-aware batching
    - Instantiates the model architecture specified in the config
    - Runs either evaluation-only or full training with periodic validation
    - Saves results and updates wandb summaries
"""

import dotenv

dotenv.load_dotenv(".env")

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import copy
import os
import random

import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import wandb
from src.constants import DATA_PATH
from src.data.dataset import BatchSampler, RNADesignDataset
from src.models import gRNAde
from src.trainer import evaluate, train


##################################
# Main Entrypoint
##################################

def main(config, device):
    """Train or evaluate gRNAde using the provided configuration.

    Args:
        config: Wandb configuration object populated from the YAML file. Expected keys include:
            - model (str): Model identifier, e.g., 'gRNAde'
            - n_samples (int): Number of sequences to sample per structure during evaluation
            - temperature (float): Sampling temperature
            - model_path (str): Optional checkpoint path to load before running
            - split (str): Data split type ('das'). Single-state only
            - device (str): 'cpu' | 'cuda'
            - num_workers (int): DataLoader workers
            - max_nodes_batch (int): Max nodes per batch
            - max_nodes_sample (int): Max nodes per single sample
            - radius, top_k, num_rbf, num_posenc, max_num_conformers, noise_scale, drop_prob_3d
        device (torch.device): Target device for computation

    Workflow:
        1. Set reproducible seeds
        2. Build model and optionally load a checkpoint
        3. If evaluate-only: construct test set and run `evaluate()`
        4. Else: build train/val/test sets and run `train()`
        5. Save results and update wandb summaries
    """
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
    print("="*80 + "\n")
    
    # Set seed
    set_seed(config.seed, device.type)

    # Initialise model
    model = get_model(config).to(device)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f"\nMODEL\n    {model}\n    Total parameters: {total_param}")
    wandb.run.summary["total_param"] = total_param

    # Load checkpoint
    if config.model_path != "":
        model.load_state_dict(torch.load(config.model_path, map_location=device))

    if config.evaluate:
        # Load test set
        _, _, test_list = get_data_splits(config, split_type=config.split)
        testset = get_dataset(config, test_list, split="test")
        test_loader = get_dataloader(config, testset, shuffle=False)

        # Run evaluator + save designed structures
        results = evaluate(
            model,
            test_loader.dataset,
            config.n_samples,
            config.temperature,
            device,
            model_name="test",
            metrics=[
                "recovery",
                "perplexity",
                "sc_score_ribonanzanet",
                "sc_score_ribonanzanet_ss",
            ],
            save_designs=False,
        )
        (
            df,
            samples_list,
            recovery_list,
            perplexity_list,
            scscore_ribonanzanet_list,
            scscore_ribonanzanet_ss_list,
        ) = results.values()
        # Save results
        torch.save(results, os.path.join(wandb.run.dir, f"test_results.pt"))
        # Update wandb summary metrics
        wandb.run.summary[f"best_test_recovery"] = np.mean(recovery_list)
        wandb.run.summary[f"best_test_perplexity"] = np.mean(perplexity_list)
        wandb.run.summary[f"best_test_scscore_ribonanzanet"] = np.mean(scscore_ribonanzanet_list)
        wandb.run.summary[f"best_test_scscore_ribonanzanet_ss"] = np.mean(
            scscore_ribonanzanet_ss_list
        )
        print(
            f"BEST test recovery: {np.mean(recovery_list):.4f}\
                perplexity: {np.mean(perplexity_list):.4f}\
                scscore_ribonanzanet: {np.mean(scscore_ribonanzanet_list):.4f}\
                scscore_ribonanzanet_ss: {np.mean(scscore_ribonanzanet_ss_list):.4f}"
        )

    else:
        # Get train, val, test data samples as lists
        train_list, val_list, test_list = get_data_splits(config, split_type=config.split)

        # Load datasets
        trainset = get_dataset(config, train_list, split="train")
        valset = get_dataset(config, val_list, split="test")  # 2D-only validation set
        testset = get_dataset(config, test_list, split="test_2d")  # 2D+3D validation set

        # Prepare dataloaders
        train_loader = get_dataloader(config, trainset, shuffle=True)
        val_loader = get_dataloader(config, valset, shuffle=False)
        test_loader = get_dataloader(config, testset, shuffle=False)

        # Run trainer
        train(config, model, train_loader, val_loader, test_loader, device)


##################################
# Data Split Helpers
##################################

def get_data_splits(config, split_type="das"):
    """Load precomputed train/val/test splits and return lists of samples.

    This function uses all the available data for training the best performing 
    gRNAde models released publicly. For evaluation, two sets are constructed 
    from the Single-State 'Das' split:
        - Validation set: 2D+3D structures from the test split
        - Test set: 2D-only structures from the test split
    
    This allows us to select the best checkpoint for real-world usage.

    For benchmarking or ablation studies, users can modify this function to
    implement alternative splitting strategies as needed.

    Args:
        config: Wandb configuration object (unused here, kept for parity)
        split_type (str): Split name. Only 'das' is supported.

    Returns:
        tuple[list, list, list]: `(train_list, val_list, test_list)`, each a list
        of raw RNA sample dictionaries from the preprocessed data.
    """
    assert split_type == "das", "Only single-state split is supported."

    data_list = list(torch.load(os.path.join(DATA_PATH, "processed.pt")).values())

    def index_list_by_indices(lst, indices):
        # return [lst[index] if 0 <= index < len(lst) else None for index in indices]
        return [lst[index] for index in indices]

    # Pre-compute using notebooks/split_{split_type}.ipynb
    train_idx_list, val_idx_list, test_idx_list = torch.load(
        os.path.join(DATA_PATH, f"{split_type}_split.pt")
    )

    # Use all data for training
    train_list = index_list_by_indices(data_list, train_idx_list)
    val_list = index_list_by_indices(data_list, val_idx_list)
    train_list = train_list + val_list

    # Two validation sets from test_idx_list:
    # (1) 2D+3D - val and (2) 2D-only - test
    val_list = index_list_by_indices(copy.deepcopy(data_list), test_idx_list)
    test_list = index_list_by_indices(copy.deepcopy(data_list), test_idx_list)

    return train_list, val_list, test_list


##################################
# Dataset & DataLoader Builders
##################################

def get_dataset(config, data_list, split="train"):
    """Construct an `RNADesignDataset` for the requested split.

    Args:
        config: Wandb configuration with featurizer and batching parameters
        data_list (list): List of raw RNA samples (dicts) to include
        split (str): One of 'train', 'test', or 'test_2d'. Controls augmentation and
            optional shuffling behavior downstream.

    Returns:
        RNADesignDataset: Dataset ready to be wrapped by a DataLoader

    Notes:
        - Only single-state evaluation is supported (`max_num_conformers == 1`).
        - `random_order` is enabled only for training to improve robustness.
    """
    assert config.max_num_conformers == 1, "Only single-state split is supported."

    return RNADesignDataset(
        data_list=data_list,
        split=split,
        radius=config.radius,
        top_k=config.top_k,
        num_rbf=config.num_rbf,
        num_posenc=config.num_posenc,
        max_num_conformers=config.max_num_conformers,
        noise_scale=config.noise_scale,
        drop_prob_3d=config.drop_prob_3d,
        random_order=(config.random_order and split == "train"),
    )


def get_dataloader(
    config,
    dataset,
    shuffle=True,
    pin_memory=True,
    exclude_keys=[],
):
    """Wrap the dataset in a memory-aware `DataLoader`.

    Args:
        config: Wandb configuration supplying loader and batching parameters
        dataset (RNADesignDataset): Dataset instance to iterate over
        shuffle (bool): Whether to shuffle batches (True for training)
        pin_memory (bool): Whether to pin memory for faster host→device transfers
        exclude_keys (list): Keys to exclude from collation (passed to torch_geometric)

    Returns:
        torch_geometric.loader.DataLoader: Configured dataloader with a `BatchSampler`
            that respects `max_nodes_batch` and `max_nodes_sample` to prevent OOM.
    """
    return DataLoader(
        dataset,
        num_workers=config.num_workers,
        batch_sampler=BatchSampler(
            node_counts=dataset.node_counts,
            max_nodes_batch=config.max_nodes_batch,
            max_nodes_sample=config.max_nodes_sample,
            shuffle=shuffle,
        ),
        pin_memory=pin_memory,
        exclude_keys=exclude_keys,
    )


def get_model(config):
    """Instantiate the model specified by the configuration.

    Supported identifiers:
        - 'gRNAde': Structure-conditioned generative RNA language model
        - (new models can be added here)

    Args:
        config: Wandb configuration specifying architecture and dimensionalities

    Returns:
        torch.nn.Module: Constructed model moved to the appropriate device by caller
    """
    model_class = {
        "gRNAde": gRNAde,
    }[config.model]

    return model_class(
        node_in_dim=tuple(config.node_in_dim),
        node_h_dim=tuple(config.node_h_dim),
        edge_in_dim=tuple(config.edge_in_dim),
        edge_h_dim=tuple(config.edge_h_dim),
        num_layers=config.num_layers,
        drop_rate=config.drop_rate,
        out_dim=config.out_dim,
    )


def set_seed(seed=0, device_type="cpu"):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", default="configs/default.yaml", type=str)
    parser.add_argument("--expt_name", dest="expt_name", default=None, type=str)
    parser.add_argument("--tags", nargs="+", dest="tags", default=[])
    parser.add_argument("--no_wandb", action="store_true")
    args, unknown = parser.parse_known_args()

    # Initialise wandb
    if args.no_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            config=args.config,
            name=args.expt_name,
            mode="disabled",
        )
    else:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            config=args.config,
            name=args.expt_name,
            tags=args.tags,
            mode="online",
        )
    config = wandb.config
    config_str = "\nCONFIG"
    for key, val in config.items():
        config_str += f"\n    {key}: {val}"
    print(config_str)

    # Set device (GPU/CPU)
    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")

    # Run main function
    main(config, device)

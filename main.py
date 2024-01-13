import dotenv
dotenv.load_dotenv(".env")

import os
import random
import argparse
import wandb
import numpy as np

from lovely_numpy import lo
import lovely_tensors as lt
lt.monkey_patch()

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from src.trainer import train, evaluate
from src.data.dataset import RNADesignDataset, BatchSampler
from src.models import (
    AutoregressiveMultiGNNv1, 
    NonAutoregressiveMultiGNNv1,
)
from src.constants import DATA_PATH

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main(config, device):
    """
    Main function for training and evaluating gRNAde.
    """
    # Set seed
    set_seed(config.seed)

    # Get train, val, test data samples as lists
    train_list, val_list, test_list = get_data_splits(config, split_type=config.split)

    # Load datasets
    trainset = get_dataset(config, train_list, split="train")
    valset = get_dataset(config, val_list, split="val")
    testset = get_dataset(config, test_list, split="test")

    # Prepare dataloaders
    train_loader = get_dataloader(config, trainset, shuffle=True)
    val_loader = get_dataloader(config, valset, shuffle=False)
    test_loader = get_dataloader(config, testset, shuffle=False)
    
    # Initialise model
    model = get_model(config).to(device)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'\nMODEL\n    {model}\n    Total parameters: {total_param}')
    wandb.run.summary["total_param"] = total_param

    # Load checkpoint
    if config.model_path != '':
        model.load_state_dict(torch.load(config.model_path))
    
    if config.evaluate:
        # val set
        val_df, val_samples_list, val_recovery_list, val_scscore_list = evaluate(
            model, 
            val_loader.dataset, 
            config.n_samples, 
            config.temperature, 
            device, 
            model_name="val"
        )
        wandb.run.summary["best_val_recovery"] = np.mean(val_recovery_list)
        wandb.run.summary["best_val_scscore"] = np.mean(val_scscore_list)
        print(f"VAL recovery: {np.mean(val_recovery_list):.4f} \
              scscore: {np.mean(val_scscore_list):.4f}")
        torch.save(
            (val_df, val_samples_list, val_recovery_list, val_scscore_list),
            os.path.join(wandb.run.dir, f"val_results.pt")
        )

        # test set
        test_df, test_samples_list, test_recovery_list, test_scscore_list = evaluate(
            model, 
            test_loader.dataset, 
            config.n_samples, 
            config.temperature, 
            device, 
            model_name="test"
        )
        wandb.run.summary["best_test_recovery"] = np.mean(test_recovery_list)
        wandb.run.summary["best_test_scscore"] = np.mean(test_scscore_list)
        print(f"TEST recovery: {np.mean(test_recovery_list):.4f} \
              scscore: {np.mean(test_scscore_list):.4f}")
        torch.save(
            (test_df, test_samples_list, test_recovery_list, test_scscore_list),
            os.path.join(wandb.run.dir, f"test_results.pt")
        )

    else:
        # Training loop
        train(config, model, train_loader, val_loader, test_loader, device)


def get_data_splits(config, split_type="structsim"):
    """
    Returns train, val, test data splits as lists.
    """
    data_list = list(torch.load(os.path.join(DATA_PATH, "processed.pt")).values())
    
    def index_list_by_indices(lst, indices):
        # return [lst[index] if 0 <= index < len(lst) else None for index in indices]
        return [lst[index] for index in indices]
    
    # Pre-compute using notebooks/split_{split_type}.ipynb
    train_idx_list, val_idx_list, test_idx_list = torch.load(
        os.path.join(DATA_PATH, f"{split_type}_split.pt")) 
    train_list = index_list_by_indices(data_list, train_idx_list)
    val_list = index_list_by_indices(data_list, val_idx_list)
    test_list = index_list_by_indices(data_list, test_idx_list)

    return train_list, val_list, test_list


def get_dataset(config, data_list, split="train"):
    """
    Returns a Dataset for a given split.
    """
    return RNADesignDataset(
        data_list = data_list,
        split = split,
        radius = config.radius,
        top_k = config.top_k,
        num_rbf = config.num_rbf,
        num_posenc = config.num_posenc,
        max_num_conformers = config.max_num_conformers,
        noise_scale = config.noise_scale
    )


def get_dataloader(
        config, 
        dataset, 
        shuffle=True,
        pin_memory=True,
        # drop_last=False,
        exclude_keys=[],
    ):
    """
    Returns a DataLoader for a given Dataset.

    Args:
        dataset (RNADesignDataset): dataset object
        config (dict): wandb configuration dictionary
        shuffle (bool): whether to shuffle the dataset
        pin_memory (bool): whether to pin memory
        # drop_last (bool): whether to drop the last batch
        exclue_keys (list): list of keys to exclude during batching
    """
    return DataLoader(
        dataset, 
        num_workers = config.num_workers,
        batch_sampler = BatchSampler(
            node_counts = dataset.node_counts, 
            max_nodes_batch = config.max_nodes_batch,
            max_nodes_sample = config.max_nodes_sample,
            shuffle = shuffle,
        ),
        pin_memory = pin_memory,
        # drop_last = drop_last,
        exclude_keys = exclude_keys
    )


def get_model(config):
    """
    Returns a Model for a given config.
    """
    model_class = {
        'ARv1' : AutoregressiveMultiGNNv1,
        'NARv1': NonAutoregressiveMultiGNNv1,
    }[config.model]
    
    return model_class(
        node_in_dim = tuple(config.node_in_dim),
        node_h_dim = tuple(config.node_h_dim), 
        edge_in_dim = tuple(config.edge_in_dim),
        edge_h_dim = tuple(config.edge_h_dim), 
        num_layers=config.num_layers,
        drop_rate = config.drop_rate,
        out_dim = config.out_dim
    )


def set_seed(seed=0):
    """
    Sets random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='configs/default.yaml', type=str)
    parser.add_argument('--expt_name', dest='expt_name', default=None, type=str)
    parser.add_argument('--tags', nargs='+', dest='tags', default=[])
    parser.add_argument('--no_wandb', action="store_true")
    args, unknown = parser.parse_known_args()

    # Initialise wandb
    if args.no_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"), 
            entity=os.environ.get("WANDB_ENTITY"), 
            config=args.config, 
            name=args.expt_name, 
            mode='disabled'
        )
    else:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"), 
            entity=os.environ.get("WANDB_ENTITY"), 
            config=args.config, 
            name=args.expt_name, 
            tags=args.tags,
            mode='online'
        )
    config = wandb.config
    config_str = "\nCONFIG"
    for key, val in config.items():
        config_str += f"\n    {key}: {val}"
    print(config_str)

    # Set device (GPU/CPU)
    device = torch.device("cuda:{}".format(config.gpu) if torch.cuda.is_available() else "cpu")
    
    # Run main function
    main(config, device)

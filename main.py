import os
import random
import argparse
import wandb
import numpy as np
from functools import partial

import torch
import torch_geometric

from src.train import train, test_recovery, test_perplexity
from src.data import RNADesignDataset, BatchSampler
from src.data_utils import process_raw, get_avg_rmsds
from src.model import AutoregressiveMultiGNN, NonAutoregressiveMultiGNN


def main(config, device):
    seed(config.seed)

    # Get train, val, test data samples as lists
    train_list, val_list, test_list = get_data_splits(config, split_type=config.split)

    # Load datasets
    trainset = get_dataset(config, train_list, split="train")
    valset = get_dataset(config, val_list, split="val")
    testset = get_dataset(config, test_list, split="test")

    # Prepare dataloaders
    train_loader = get_dataloader(trainset, config, shuffle=True)
    val_loader = get_dataloader(valset, config, shuffle=False)
    test_loader = get_dataloader(testset, config, shuffle=False)
    
    # Initialise model
    model = get_model(config).to(device)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    # print(model)
    print(f'Model: {type(model).__name__}, Parameters: {total_param}')

    # Load checkpoint
    if config.model_path != '':
        model.load_state_dict(torch.load(config.model_path))
    
    if config.test_recovery:
        # Evaluate recovery on test set
        test_recovery(model, testset, config.n_samples, device)
    elif config.test_perplexity:
        # Evaluate perplexity on test set
        test_perplexity(model, test_loader, device)
    else:
        # Training loop
        train(config, model, train_loader, val_loader, test_loader, device)


def get_data_splits(config, split_type="rmsd"):
    if config.process_raw == True:
        data_list = process_raw(config.data_path, config.save_processed)
    else:
        data_list = torch.load(os.path.join(config.data_path, "processed.pt"))
    
    def index_list_by_indices(lst, indices):
        # return [lst[index] if 0 <= index < len(lst) else None for index in indices]
        return [lst[index] for index in indices]
    
    # Pre-compute using notebooks/data_splitting.ipynb
    train_idx_list, val_idx_list, test_idx_list = torch.load(os.path.join(config.data_path, f"{split_type}_split.pt"))    
    train_list = index_list_by_indices(data_list, train_idx_list)
    val_list = index_list_by_indices(data_list, val_idx_list)
    test_list = index_list_by_indices(data_list, test_idx_list)
    
    return train_list, val_list, test_list


def get_dataset(config, data_list, split="train"):
    return RNADesignDataset(
        data_list = data_list,
        split = split,
        radius = config.radius,
        top_k = config.top_k,
        num_rbf = config.num_rbf,
        num_posenc = config.num_posenc,
        num_conformers = config.num_conformers,
    )


def get_dataloader(dataset, config, shuffle=True):
    return torch_geometric.loader.DataLoader(
        dataset, 
        num_workers = config.num_workers,
        batch_sampler = BatchSampler(
            node_counts = dataset.node_counts, 
            max_nodes = config.max_nodes,
            shuffle = shuffle,
        )
    )


def get_model(config):
    return {
        'AR' : AutoregressiveMultiGNN,
        'NAR': NonAutoregressiveMultiGNN,
    }[config.model](
        node_in_dim = tuple(config.node_in_dim),
        node_h_dim = tuple(config.node_h_dim), 
        edge_in_dim = tuple(config.edge_in_dim),
        edge_h_dim = tuple(config.edge_h_dim), 
        num_layers=config.num_layers,
        drop_rate = config.drop_rate,
        out_dim = config.out_dim
    )


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


print = partial(print, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='configs/default.yaml', type=str)
    parser.add_argument('--expt_name', dest='expt_name', default=None, type=str)
    parser.add_argument('--no_wandb', action="store_true")
    args, unknown = parser.parse_known_args()

    # Initialise wandb
    if args.no_wandb:
        wandb.init(project="gRNAde", entity="chaitjo", config=args.config, name=args.expt_name, mode='disabled')
    else:
        wandb.init(project="gRNAde", entity="chaitjo", config=args.config, name=args.expt_name, mode='online')
    config = wandb.config
    for key, val in config.items():
        print(f"  {key}: {val}")

    # Set device (GPU/CPU)
    device = torch.device("cuda:{}".format(config.gpu) if torch.cuda.is_available() else "cpu")
    
    main(config, device)

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix
from torchmetrics.functional.classification import binary_matthews_corrcoef

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.data_utils import get_c4p_coords
from src.data.sec_struct_utils import (
    predict_sec_struct,
    dotbracket_to_paired,
    dotbracket_to_adjacency
)
from src.constants import NUM_TO_LETTER


def train(
        config, 
        model, 
        train_loader, 
        val_loader, 
        test_loader, 
        device
    ):
    """
    Train RNA inverse folding model using the specified config and data loaders.
    """

    # Initialise loss function
    train_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    eval_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    
    # Initialise optimizer and scheduler
    lr = config.lr
    optimizer = Adam(model.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=1, min_lr=0.00001)
    
    # Initialise lookup table mapping integers to nucleotides
    lookup = train_loader.dataset.featurizer.num_to_letter
    
    # Initialise best checkpoint information
    best_epoch, best_val_loss, best_val_acc = -1, np.inf, 0
    
    # Training loop
    for epoch in range(config.epochs):
        
        # Training iteration
        model.train()
        train_loss, train_acc, train_confusion = loop(model, train_loader, train_loss_fn, optimizer, device)
        print_and_log(epoch, train_loss, train_acc, train_confusion, lr=lr, mode="train", lookup=lookup)
        
        if epoch % config.val_every == 0 or epoch == config.epochs - 1:
            
            model.eval()
            with torch.no_grad(): 
                
                # Evaluate on validation set
                val_loss, val_acc, val_confusion = loop(model, val_loader, eval_loss_fn, None, device)
                print_and_log(epoch, val_loss, val_acc, val_confusion, mode="val", lookup=lookup)

                # LR scheduler step
                scheduler.step(val_acc)
                lr = optimizer.param_groups[0]['lr']
                
                if val_acc > best_val_acc:
                    # Update best checkpoint
                    best_epoch, best_val_loss, best_val_acc = epoch, val_loss, val_acc

                    # Evaluate on test set
                    test_loss, test_acc, test_confusion = loop(model, test_loader, eval_loss_fn, None, device)
                    print_and_log(epoch, test_loss, test_acc, test_confusion, mode="test", lookup=lookup)

                    # Update wandb summary metrics
                    wandb.run.summary["best_epoch"] = best_epoch
                    wandb.run.summary["best_val_perp"] = np.exp(best_val_loss)
                    wandb.run.summary["best_val_acc"] = best_val_acc
                    wandb.run.summary["best_test_perp"] = np.exp(test_loss)
                    wandb.run.summary["best_test_acc"] = test_acc
                    
                    if config.save:
                        # Save best checkpoint
                        checkpoint_path = os.path.join(wandb.run.dir, "best_checkpoint.h5")
                        torch.save(model.state_dict(), checkpoint_path)
                        wandb.run.summary["best_checkpoint"] = checkpoint_path

        if config.save:
            # Save current epoch checkpoint
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "current_checkpoint.h5"))
    
    # End of training
    if config.save:
        # Evaluate best checkpoint
        print(f"EVALUATION: loading {os.path.join(wandb.run.dir, 'best_checkpoint.h5')} (epoch {best_epoch})")
        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, 'best_checkpoint.h5')))

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
        print(f"BEST VAL recovery: {np.mean(val_recovery_list):.4f} scscore: {np.mean(val_scscore_list):.4f}")
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
        print(f"BEST TEST recovery: {np.mean(test_recovery_list):.4f} scscore: {np.mean(test_scscore_list):.4f}")
        torch.save(
            (test_df, test_samples_list, test_recovery_list, test_scscore_list),
            os.path.join(wandb.run.dir, f"test_results.pt")
        )


def loop(model, dataloader, loss_fn, optimizer=None, device='cpu'):
    """
    Training loop for a single epoch.
    """

    confusion = np.zeros((model.out_dim, model.out_dim))
    total_loss, total_correct, total_count = 0, 0, 0
    
    t = tqdm(dataloader)
    for batch in t:
        if optimizer: optimizer.zero_grad()
    
        # move batch to device
        batch = batch.to(device)
        
        try:
            logits = model(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise(e)
            print('Skipped batch due to OOM', flush=True)
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            torch.cuda.empty_cache()
            continue
        
        # compute loss
        loss_value = loss_fn(logits, batch.seq)
        
        if optimizer:
            # backpropagate loss and update parameters
            loss_value.backward()
            optimizer.step()

        # update metrics
        num_nodes = int(batch.seq.size(0))
        total_loss += float(loss_value.item()) * num_nodes
        total_count += num_nodes
        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        true = batch.seq.detach().cpu().numpy()
        total_correct += (pred == true).sum()
        confusion += confusion_matrix(true, pred, labels=range(model.out_dim))
        
        t.set_description("%.5f" % float(total_loss/total_count))
        
    return total_loss / total_count, total_correct / total_count, confusion


def evaluate(
        model, 
        dataset, 
        n_samples, 
        temperature, 
        device, 
        model_name="eval"
    ):
    """
    Run evaluation suite for trained RNA inverse folding model on a dataset.

    The following metrics are computed:
    1. Sequence recovery per residue (taking mean gives per sample recovery)
    2. Self consistency score per sample, based on predicted secondary structure
    ...along with metadata per sample per residue.

    Args:
        model: trained RNA inverse folding model
        dataset: dataset to evaluate on
        n_samples: number of predicted samples/sequences per data point 
        temperature: sampling temperature
        device: device to run evaluation on
        model_name: name of model/dataset for plotting
    """

    # per sample metric lists
    samples_list = []   # list of tensors of shape (n_samples, seq_len) per data point 
    recovery_list = []  # list of mean recovery per data point
    sc_score_list = []  # list of mean self-consistency scores per data point

    # DataFrame to store metrics and metadata per residue per sample for analysis and plotting
    df = pd.DataFrame(columns=['idx', 'recovery', 'sasa', 'paired', 'rmsds', 'model_name'])

    model.eval()
    with torch.no_grad():
        for idx, raw_data in tqdm(
            enumerate(dataset.data_list),
            total=len(dataset.data_list)
        ):
            # featurise raw data
            data = dataset.featurizer(raw_data).to(device)

            # sample n_samples from model for single data point: n_samples x seq_len
            samples = model.sample(data, n_samples, temperature)

            ###########
            # Metadata
            ###########

            # per residue average SASA: seq_len x 1
            mask_coords = data.mask_coords.cpu().numpy()
            sasa = np.mean(raw_data['sasa_list'], axis=0)[mask_coords]

            # per residue indicator for paired/unpaired: seq_len x 1
            paired = np.mean(
                [dotbracket_to_paired(sec_struct) for sec_struct in raw_data['sec_struct_list']], axis=0
            )[mask_coords]

            # per residue average RMSD: seq_len x 1
            if len(raw_data["coords_list"]) == 1:
                rmsds = np.zeros_like(sasa)
            else:
                rmsds = []
                for i in range(len(raw_data["coords_list"])):
                    for j in range(i+1, len(raw_data["coords_list"])):
                        coords_i = get_c4p_coords(raw_data["coords_list"][i])
                        coords_j = get_c4p_coords(raw_data["coords_list"][j])
                        rmsds.append(torch.sqrt(torch.sum((coords_i - coords_j)**2, dim=1)).cpu().numpy())
                rmsds = np.stack(rmsds).mean(axis=0)[mask_coords]

            ##########
            # Metrics
            ##########

            # sequence recovery per residue across all samples: seq_len x 1 
            recovery = samples.eq(data.seq).float().mean(dim=0).cpu().numpy()

            # global self consistency score per sample: n_samples x 1
            sc_score = self_consistency_score(
                samples.cpu().numpy(), 
                raw_data['sec_struct_list'], 
                mask_coords
            )

            # update per residue per sample dataframe
            df = pd.concat([
                df, 
                pd.DataFrame({
                    'idx': [idx] * len(recovery),
                    'recovery': recovery,
                    'sasa': sasa,
                    'paired': paired,
                    'rmsds': rmsds,
                    'model_name': [model_name] * len(recovery)
                })
            ], ignore_index=True)

            # update per sample lists
            samples_list.append(samples.cpu().numpy())
            recovery_list.append(recovery.mean())
            sc_score_list.append(sc_score.mean())
    
    return df, samples_list, recovery_list, sc_score_list


def self_consistency_score(
        samples, 
        true_sec_struct_list, 
        mask_coords,
        n_samples_ss = 1,
        num_to_letter = NUM_TO_LETTER,
    ):
    """
    Compute self consistency score for an RNA, given its true secondary structure(s)
    and a list of designed sequences. EternaFold is used to 'forward fold' the designs.
    
    Args:
        samples: designed sequences of shape (n_samples, seq_len)
        true_sec_struct_list: list of true secondary structures (n_true_ss, seq_len)
        mask_coords: mask for missing sequence coordinates to be ignored during evaluation
        n_samples_ss: number of predicted secondary structures per designed sample
        num_to_letter: lookup table mapping integers to nucleotides
    
    Workflow:
        
        Input: For a given RNA molecule, we are given:
            Designed sequences of shape (n_samples, seq_len)
            True secondary structure(s) of shape (n_true_ss, seq_len)
        
        For each designed sequence:
            
            Predict n_sample_ss secondary structures using EternaFold
            
            For each pair of true and predicted secondary structures:
                Compute MCC score between them
            
            Take the average MCC score across all n_sample_ss predicted structures
        
        Take the average MCC score across all n_samples designed sequences
    """
    
    n_true_ss = len(true_sec_struct_list)
    sequence_length = mask_coords.sum()
    # map all entries from dotbracket to numerical representation
    true_sec_struct_list = np.array([dotbracket_to_adjacency(ss) for ss in true_sec_struct_list])
    # mask out missing sequence coordinates
    true_sec_struct_list = true_sec_struct_list[:, mask_coords][:, :, mask_coords]
    # reshape to (n_true_ss * n_samples_ss, seq_len, seq_len)
    true_sec_struct_list = torch.tensor(
        true_sec_struct_list
    ).unsqueeze(1).repeat(1, n_samples_ss, 1, 1).reshape(-1, sequence_length, sequence_length)

    mcc_scores = []
    for _sample in samples:
        # convert sample to string
        pred_seq = ''.join([num_to_letter[num] for num in _sample])
        # predict secondary structure(s) for each sample
        pred_sec_struct_list = predict_sec_struct(pred_seq, n_samples=n_samples_ss)
        # map all entries from dotbracket to numerical representation
        pred_sec_struct_list = np.array([dotbracket_to_adjacency(ss) for ss in pred_sec_struct_list])
        # reshape to (n_samples_ss * n_true_ss, seq_len, seq_len)
        pred_sec_struct_list = torch.tensor(
            pred_sec_struct_list
        ).unsqueeze(0).repeat(n_true_ss, 1, 1, 1).reshape(-1, sequence_length, sequence_length)

        # compute mean MCC score between pairs of true and predicted secondary structures
        mcc_scores.append(
            binary_matthews_corrcoef(
                pred_sec_struct_list,
                true_sec_struct_list,
            ).float().mean()
        )

    return np.array(mcc_scores)


def print_and_log(
        epoch,
        loss,
        acc,
        confusion,
        recovery = None,
        lr = None,
        mode = "train",
        lookup = NUM_TO_LETTER, # reverse of {'A': 0, 'G': 1, 'C': 2, 'U': 3}
    ):
    # Create log string and wandb metrics dict
    log_str = f"\nEPOCH {epoch} {mode.upper()} loss: {loss:.4f} perp: {np.exp(loss):.4f} acc: {acc:.4f}"
    wandb_metrics = {
        f"{mode}/loss": loss, 
        f"{mode}/perp": np.exp(loss), 
        f"{mode}/acc": acc, 
        "epoch": epoch 
    }

    if lr is not None:
        # Add learning rate to loggers
        log_str += f" lr: {lr:.6f}"
        wandb_metrics[f"lr"] = lr

    if recovery is not None:
        # Add mean sequence recovery to loggers
        log_str += f" rec: {np.mean(recovery):.4f}"
        wandb_metrics[f"{mode}/recovery"] = np.mean(recovery)
    
    print(log_str)
    print_confusion(confusion, lookup=lookup)
    wandb.log(wandb_metrics)


def print_confusion(mat, lookup):
    counts = mat.astype(np.int32)
    mat = (counts.T / counts.sum(axis=-1, keepdims=True).T).T
    mat = np.round(mat * 1000).astype(np.int32)
    res = '\n'
    for i in range(len(lookup.keys())):
        res += '\t{}'.format(lookup[i])
    res += '\tCount\n'
    for i in range(len(lookup.keys())):
        res += '{}\t'.format(lookup[i])
        res += '\t'.join('{}'.format(n) for n in mat[i])
        res += '\t{}\n'.format(sum(counts[i]))
    print(res)

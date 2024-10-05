import os
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.evaluator import evaluate
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

    Args:
        config (dict): wandb configuration dictionary 
        model (nn.Module): RNA inverse folding model to be trained
        train_loader (DataLoader): training data loader
        val_loader (DataLoader): validation data loader
        test_loader (DataLoader): test data loader
        device (torch.device): device to train the model on
    """

    # Initialise loss function
    train_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    eval_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    
    # Initialise optimizer and scheduler
    lr = config.lr
    optimizer = Adam(model.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=1, min_lr=0.00001)

    if device.type == 'xpu':
        import intel_extension_for_pytorch as ipex
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    
    # Initialise lookup table mapping integers to nucleotides
    lookup = train_loader.dataset.featurizer.num_to_letter
    
    # Initialise best checkpoint information
    best_epoch, best_val_loss, best_val_acc = -1, np.inf, 0
    
    ##################################
    # Training loop over mini-batches
    ##################################

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
        
        for loader, set_name in [(test_loader, "test"), (val_loader, "val")]:
            # Run evaluator
            results = evaluate(
                model, 
                loader.dataset, 
                config.n_samples, 
                config.temperature, 
                device, 
                model_name=set_name,
                metrics=['recovery', 'perplexity', 'sc_score_eternafold', 'sc_score_ribonanzanet', 'sc_score_rhofold'],
                save_designs=True
            )
            df, samples_list, recovery_list, perplexity_list, \
            scscore_list, scscore_ribonanza_list, \
            scscore_rmsd_list, scscore_tm_list, scscore_gdt_list, \
            rmsd_within_thresh, tm_within_thresh, gdt_within_thresh = results.values()
            # Save results
            torch.save(results, os.path.join(wandb.run.dir, f"{set_name}_results.pt"))
            # Update wandb summary metrics
            wandb.run.summary[f"best_{set_name}_recovery"] = np.mean(recovery_list)
            wandb.run.summary[f"best_{set_name}_perplexity"] = np.mean(perplexity_list)
            wandb.run.summary[f"best_{set_name}_scscore"] = np.mean(scscore_list)
            wandb.run.summary[f"best_{set_name}_scscore_ribonanza"] = np.mean(scscore_ribonanza_list)
            wandb.run.summary[f"best_{set_name}_scscore_rmsd"] = np.mean(scscore_rmsd_list)
            wandb.run.summary[f"best_{set_name}_scscore_tm"] = np.mean(scscore_tm_list)
            wandb.run.summary[f"best_{set_name}_scscore_gdt"] = np.mean(scscore_gdt_list)
            wandb.run.summary[f"best_{set_name}_rmsd_within_thresh"] = np.mean(rmsd_within_thresh)
            wandb.run.summary[f"best_{set_name}_tm_within_thresh"] = np.mean(tm_within_thresh)
            wandb.run.summary[f"best_{set_name}_gdt_within_thresh"] = np.mean(gdt_within_thresh)
            print(f"BEST {set_name} recovery: {np.mean(recovery_list):.4f}\
                    perplexity: {np.mean(perplexity_list):.4f}\
                    scscore: {np.mean(scscore_list):.4f}\
                    scscore_ribonanza: {np.mean(scscore_ribonanza_list):.4f}\
                    scscore_rmsd: {np.mean(scscore_rmsd_list):.4f}\
                    scscore_tm: {np.mean(scscore_tm_list):.4f}\
                    scscore_gdt: {np.mean(scscore_gdt_list):.4f}\
                    rmsd_within_thresh: {np.mean(rmsd_within_thresh):.4f}\
                    tm_within_thresh: {np.mean(tm_within_thresh):.4f}\
                    gdt_within_thresh: {np.mean(gdt_within_thresh):.4f}")


def loop(model, dataloader, loss_fn, optimizer=None, device='cpu'):
    """
    Training loop for a single epoch over the data loader.

    Args:
        model (nn.Module): RNA inverse folding model
        dataloader (DataLoader): data loader for the current epoch
        loss_fn (nn.Module): loss function to compute the loss
        optimizer (torch.optim): optimizer to update model parameters
        device (torch.device): device to train the model on
    
    Note:
        This function is used for both training and evaluation loops.
        Not passing an optimizer will run the model in evaluation mode.

    Returns:
        float: average loss over the epoch
        float: average accuracy over the epoch
        np.ndarray: confusion matrix over the epoch
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

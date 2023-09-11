import os
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(config, model, train_loader, val_loader, test_loader, device):
    lr = config.lr
    optimizer = Adam(model.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=1, min_lr=0.00001)
    lookup = train_loader.dataset.num_to_letter
    best_epoch, best_val_loss, best_val_acc = -1, np.inf, 0
    # Training loop
    for epoch in range(config.epochs):
        # Training iteration
        model.train()
        train_loss, train_acc, train_confusion = loop(model, train_loader, optimizer, device)
        print(f'\nEPOCH {epoch} TRAIN loss: {train_loss:.4f} perp: {np.exp(train_loss):.4f} acc: {train_acc:.4f} lr: {lr:.6f}')
        print_confusion(train_confusion, lookup=lookup)
        wandb.log({ "train_loss": train_loss, "train_perp": np.exp(train_loss), "train_acc": train_acc, "epoch": epoch, "lr": lr })

        if epoch % config.val_every == 0:
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_loss, val_acc, val_confusion = loop(model, val_loader, None, device)
                print(f'\nEPOCH {epoch} VAL loss: {val_loss:.4f} perp: {np.exp(val_loss):.4f} acc: {val_acc:.4f}')
                print_confusion(val_confusion, lookup=lookup)
                wandb.log({ "val_loss": val_loss, "val_perp": np.exp(val_loss),  "val_acc": val_acc, "epoch": epoch })

                # LR scheduler step
                scheduler.step(val_acc)
                lr = optimizer.param_groups[0]['lr']
                
                if val_acc > best_val_acc:
                    # Update best checkpoint
                    best_epoch, best_val_loss, best_val_acc = epoch, val_loss, val_acc
                    if config.save:
                        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best_checkpoint.h5"))
                    
                    # Evaluate on test set
                    test_loss, test_acc, test_confusion = loop(model, test_loader, None, device)
                    print(f'\nEPOCH {epoch} TEST loss: {test_loss:.4f} perp: {np.exp(test_loss):.4f} acc: {test_acc:.4f}')
                    print_confusion(test_confusion, lookup=lookup)
                    wandb.log({ "test_loss": test_loss, "test_perp": np.exp(test_loss), "test_acc": test_acc, "epoch": epoch })

        if config.save:
            # Save current epoch checkpoint
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "current_checkpoint.h5"))
    
    # End of training
    print(f'BEST EPOCH {best_epoch} VAL loss: {best_val_loss:.4f} perp: {np.exp(best_val_loss):.4f} acc: {best_val_acc:.4f}')
    if config.save:
        # Evaluate best checkpoint on test set
        print(f"EVALUATION: loading from {os.path.join(wandb.run.dir, 'best_checkpoint.h5')}")
        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, 'best_checkpoint.h5')))
        model.eval()
        with torch.no_grad():
            # test_loss, test_acc, test_confusion = loop(model, test_loader, None, device)
            # print(f'BEST EPOCH {best_epoch} TEST loss: {test_loss:.4f} perp: {np.exp(test_loss):.4f} acc: {test_acc:.4f}')
            # print_confusion(test_confusion, lookup=lookup)

            # Evaluate recovery on validation set
            recovery = test_recovery(model, val_loader.dataset, config.n_samples, device)
            wandb.log({ "val_recovery_median": np.median(recovery), "val_recovery_mean": np.mean(recovery), "val_recovery_std": np.std(recovery), "epoch": best_epoch })
            torch.save(recovery, os.path.join(wandb.run.dir, "val_recovery.h5"))
            
            # Evaluate recovery on test set
            recovery = test_recovery(model, test_loader.dataset, config.n_samples, device)
            wandb.log({ "test_recovery_median": np.median(recovery), "test_recovery_mean": np.mean(recovery), "test_recovery_std": np.std(recovery), "epoch": best_epoch })
            torch.save(recovery, os.path.join(wandb.run.dir, "test_recovery.h5"))


def loop(model, dataloader, optimizer=None, device='cpu'):
    confusion = np.zeros((model.out_dim, model.out_dim))
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss, total_correct, total_count = 0, 0, 0
    
    t = tqdm(dataloader)
    for batch in t:
        if optimizer: optimizer.zero_grad()
    
        batch = batch.to(device)
        try:
            logits = model(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise(e)
            print('Skipped batch due to OOM', flush=True)
            torch.cuda.empty_cache()
            continue
        
        logits, seq = logits[batch.mask_coords], batch.seq[batch.mask_coords]
        loss_value = loss_fn(logits, seq)

        if optimizer:
            loss_value.backward()
            optimizer.step()

        num_nodes = int(batch.mask_coords.sum())
        total_loss += float(loss_value.item()) * num_nodes
        total_count += num_nodes
        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        true = seq.detach().cpu().numpy()
        total_correct += (pred == true).sum()
        confusion += confusion_matrix(true, pred, labels=range(model.out_dim))
        
        t.set_description("%.5f" % float(total_loss/total_count))
        
    return total_loss / total_count, total_correct / total_count, confusion


def test_perplexity(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        loss, acc, confusion = loop(model, dataloader, None, device)
    print(f'EVAL perplexity: {np.exp(loss):.4f}')
    print_confusion(confusion, lookup=dataloader.dataset.num_to_letter)
    return np.exp(loss)


def test_recovery(model, dataset, n_samples, device):
    recovery = []
    t = tqdm(dataset)
    for data in t:
        data = data.to(device)   
        sample = model.sample(data, n_samples=n_samples)
        recovery_ = sample.eq(data.seq).float().mean().cpu().numpy()
        recovery.append(recovery_)
        t.set_description(f"{np.median(recovery):.4f}")
    print(f'EVAL recovery median: {np.median(recovery):.4f} mean: {np.mean(recovery):.4f} +- {np.std(recovery):.4f}')
    return recovery


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

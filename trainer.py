import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path
from models import ParticleNet, JetGAT

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(name):
    if name == 'particlenet':
        return ParticleNet(
            in_channels   = 6,
            k             = 16,
            edge_channels = (64, 128, 256),
            fc_channels   = 256,
            dropout       = 0.1
        )
    elif name == 'jetgat':
        return JetGAT(
            in_channels  = 6,
            hidden_dim   = 64,
            heads        = 4,
            num_layers   = 3,
            fc_channels  = 256,
            dropout      = 0.1,
            edge_dim     = 4
        )
    raise ValueError(f'Unknown model: {name}')

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def lr_schedule(optimizer, epoch, warmup, total, base_lr):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / (total - warmup)
        lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr

def run_epoch(model, loader, optimizer, criterion, scaler, training):
    model.train(training)
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(DEVICE)
            with torch.amp.autocast(device_type='cuda',
                                     enabled=(DEVICE.type == 'cuda')):
                logits = model(batch)
                loss   = criterion(logits, batch.y.squeeze())

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * batch.num_graphs
            probs = torch.softmax(logits.detach(), dim=-1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(batch.y.squeeze().cpu().numpy())

    n   = len(all_labels)
    acc = ((np.array(all_preds) > 0.5).astype(int) == np.array(all_labels)).mean()
    auc = roc_auc_score(all_labels, all_preds)
    return total_loss / n, acc, auc

def train_model(
    model_name,
    train_data, val_data, test_data,
    ckpt_dir, log_dir,
    epochs        = 30,
    batch_size    = 128,
    lr            = 1e-3,
    patience      = 7,
    warmup_epochs = 3,
):
    ckpt_dir    = Path(ckpt_dir)
    log_dir     = Path(log_dir)
    ckpt_path   = ckpt_dir / f'{model_name}_best.pt'
    resume_path = ckpt_dir / f'{model_name}_resume.pt'
    log_path    = log_dir  / f'{model_name}_history.json'

    train_loader = DataLoader(
        train_data, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size * 2,
        shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size * 2,
        shuffle=False, num_workers=2, pin_memory=True
    )

    model     = build_model(model_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    start_epoch  = 0
    best_val_auc = 0.0
    patience_ctr = 0
    history      = []

    if resume_path.exists():
        print(f'[trainer] Resuming from {resume_path}')
        ckpt = torch.load(resume_path, map_location=DEVICE,
                          weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch  = ckpt['epoch']
        best_val_auc = ckpt['best_val_auc']
        patience_ctr = ckpt['patience_ctr']
        history      = ckpt['history']
        print(f'[trainer] Resumed at epoch {start_epoch + 1}  '
              f'best AUC so far = {best_val_auc:.4f}')
    else:
        print(f'[trainer] Starting fresh')
        print(f'[trainer] Parameters : {count_params(model):,}')

    print(f'[trainer] Device     : {DEVICE}')
    print(f'[trainer] Epochs     : {start_epoch + 1} → {epochs}')
    print(f'[trainer] Batch size : {batch_size}')
    print('=' * 65)

    for epoch in range(start_epoch, epochs):
        t0     = time.time()
        lr_now = lr_schedule(optimizer, epoch, warmup_epochs, epochs, lr)

        tr_loss, tr_acc, tr_auc = run_epoch(
            model, train_loader, optimizer, criterion, scaler, training=True)
        vl_loss, vl_acc, vl_auc = run_epoch(
            model, val_loader, optimizer, criterion, scaler, training=False)

        dt = time.time() - t0
        print(f'Ep {epoch+1:>3}/{epochs}  '
              f'lr={lr_now:.2e}  '
              f'train[loss={tr_loss:.4f} acc={tr_acc:.4f} auc={tr_auc:.4f}]  '
              f'val[loss={vl_loss:.4f} acc={vl_acc:.4f} auc={vl_auc:.4f}]  '
              f'{dt:.0f}s')

        history.append(dict(
            epoch=epoch + 1,     lr=lr_now,
            train_loss=tr_loss,  train_acc=tr_acc,  train_auc=tr_auc,
            val_loss=vl_loss,    val_acc=vl_acc,    val_auc=vl_auc,
        ))

        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            patience_ctr = 0
            torch.save({
                'epoch':       epoch + 1,
                'model_state': model.state_dict(),
                'val_auc':     vl_auc,
            }, ckpt_path)
            print(f'  ✔ Best model saved  (val AUC={best_val_auc:.4f})')
        else:
            patience_ctr += 1

        torch.save({
            'epoch':           epoch + 1,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state':    scaler.state_dict(),
            'best_val_auc':    best_val_auc,
            'patience_ctr':    patience_ctr,
            'history':         history,
        }, resume_path)

        with open(log_path, 'w') as f:
            json.dump({'history': history}, f, indent=2)

        if patience_ctr >= patience:
            print(f'\n  Early stopping — no improvement for {patience} epochs.')
            break

    print('\n' + '-' * 65)
    print('Loading best checkpoint for test evaluation ...')
    best_ckpt = torch.load(ckpt_path, map_location=DEVICE,
                           weights_only=False)
    model.load_state_dict(best_ckpt['model_state'])

    te_loss, te_acc, te_auc = run_epoch(
        model, test_loader, optimizer, criterion, scaler, training=False)

    print(f'\nTEST  loss={te_loss:.4f}  acc={te_acc:.4f}  AUC={te_auc:.4f}')
    print('-' * 65)

    with open(log_path, 'w') as f:
        json.dump({
            'history':   history,
            'test_loss': te_loss,
            'test_acc':  te_acc,
            'test_auc':  te_auc,
        }, f, indent=2)

    print(f'[trainer] Log saved → {log_path}')
    return history, te_auc

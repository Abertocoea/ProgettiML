
from models.early_stopping import EarlyStopping
import torch
import numpy as np

def train_neural_network(model, train_loader, vali_loader, n_epochs, patience, verbose):
    train_loss_list = []
    val_loss_list = []
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)

    for epoch in range(n_epochs):
        batch_train_loss = []
        batch_val_loss = []
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_func(yb, pred)
            loss.backward()
            opt.step()
            batch_train_loss.append(loss.item())

        train_loss_list.append(np.mean(batch_train_loss))
        model.eval()
        with torch.no_grad():
            for xb, yb in vali_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_func(yb, pred)
                batch_val_loss.append(loss.item())

            val_loss_list.append(np.mean(batch_val_loss))

        early_stopping(val_loss_list[epoch], model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch: {epoch}")
            break
        print(f'Epoch {epoch} -- train_loss={train_loss_list[epoch]:.4f} -- val_loss={val_loss_list[epoch]:.4f}')

    return train_loss_list, val_loss_list
# -*- coding: utf-8 -*-
import os
import pickle
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

def preprocessing(input, target, perc_test, shuffle, nn_or_ml, scaler_path="utils/scaler.pkl"):
    # Creazione della cartella se non esiste
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    # Split dei dati
    x_train, x_test, y_train, y_test = train_test_split(input, target, test_size=perc_test, random_state=1234, shuffle=shuffle)

    # Normalizzazione dei dati
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Salvataggio dello scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    if nn_or_ml:  # True: dataset PyTorch
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        return train_dataset, test_dataset
    else:  # False: ritorna lo split normale
        return x_train, x_test, y_train, y_test

def load_scaler(scaler_path="utils/scaler.pkl"):

    with open(scaler_path, 'rb') as f:
        return pickle.load(f)
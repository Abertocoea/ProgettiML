
from train import train_neural_network
from models.ffnn import FFNN
from utils.preprocessing import preprocessing
from utils.plotting import plot_history
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Configurazione dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Computation device: {device}\n")

# Caricamento e preprocessing dei dati
perc_test = 0.2  # Percentuale di dati per il test
shuffle = True
nn_or_ml = True  # True per dataset PyTorch, False per array numpy

train_dataset, test_dataset = preprocessing(
    input, target, perc_test, shuffle, nn_or_ml, save_scaler_path="utils/scaler.pkl"
)

# Creazione dei DataLoader per PyTorch
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
vali_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modello
input_dim = train_dataset.tensors[0].shape[1]
output_size = train_dataset.tensors[1].shape[1] if len(train_dataset.tensors[1].shape) > 1 else 1

model = FFNN(input_dim, output_size, 128, 3, 0.0).to(device)
loss_func = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=0.001)

# Training
t0 = time.time()
train_loss_list, val_loss_list = train_neural_network(model, train_loader, vali_loader, n_epochs=50, patience=10, verbose=True)
t1 = time.time()
print(f'Tempo di esecuzione: {t1-t0}')

# Plot storico perdita
plot_history(train_loss_list, val_loss_list, '3R robot train/val same seed')

# Caricamento miglior modello
model.load_state_dict(torch.load('checkpoints/checkpoint.pt'))
model.to('cpu')

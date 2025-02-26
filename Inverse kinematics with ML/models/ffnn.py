
import torch.nn as nn

class FFNN(nn.Module):
    def __init__(self, input_size, output_size, depth=128, num_layers=3, p=0.3):
        super(FFNN, self).__init__()
        self.dropout = nn.Dropout(p)
        self.num_layers = num_layers
        self.depth = depth
        self.input_layer = nn.Linear(input_size, depth)
        self.hidden_layer = nn.Linear(depth, depth)
        self.output_layer = nn.Linear(depth, output_size)
        self.relu = nn.ReLU()
        layers = [self.input_layer, self.relu, self.dropout]
        for _ in range(self.num_layers - 2):
            layers += [self.hidden_layer, self.relu, self.dropout]
        layers.append(self.output_layer)
        self.ffnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffnn(x)
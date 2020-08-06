import torch
import torch.nn as nn


class Fnn(nn.Module):
    def __init__(self, ip_size, hidden_size, num_classes):
        super().__init__()
        self.ip = ip_size
        self.linear = nn.Linear(ip_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape(-1, self.ip)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

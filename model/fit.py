import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST

data = MNIST(root="./data", download=True)
print(data)
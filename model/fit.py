import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from model import Fnn

data = MNIST(root="./data", download=True)
# about the data
print(data)

data1 = MNIST(root="./data",
              download=False,
              transform=transforms.ToTensor(),
              train=True)
i, j = data1[0]
mean = []
std = []
for x in range(len(i)):
    mean.append(i[x].mean())
    std.append(i[x].std())
# mean and std for normalization
print(mean)
print(std)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)])

data = MNIST(root="./data", download=False, transform=transform, train=True)

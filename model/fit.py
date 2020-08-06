import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from model import Fnn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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

train, vali = torch.utils.data.random_split(data, [50000, 10000])

dl = DataLoader(train, batch_size=64, shuffle=True)
vl = DataLoader(vali, batch_size=64, shuffle=True)

model = Fnn(784, 64, 10)

for i, j in dl:
    pred = model(i)
    break


def accuracy(pred, label):
    _, pred = torch.max(pred, dim=1)
    return torch.tensor(torch.sum(pred == label).item() / len(label))


loss = nn.CrossEntropyLoss()
print(loss(pred, j))
print(accuracy(pred, j))


def fit(epochs, lr, model, loss, dl, vl):
    opt = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for i, j in dl:
            pred = model(i)
            ls = loss(pred, j)
            ls.backward()
            opt.step()

        for i, j in vl:
            pred = model(i)
            ls = loss(pred, j)
            acc = accuracy(pred, j)
            print(f"epoch.no={epoch+1} loss={ls} acc={acc}")


epochs = 20
lr = 2e-5
model = Fnn(784, 64, 10)
loss = nn.CrossEntropyLoss()

fit(epochs, lr, model, loss, dl, vl)

torch.save(model.state_dict(), "fnn.pth")
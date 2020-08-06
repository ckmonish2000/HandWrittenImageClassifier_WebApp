import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from model.model import Fnn
from torchvision.utils import make_grid
import io
from PIL import Image

model = Fnn(784, 64, 10)
model.load_state_dict(torch.load("./model/fnn.pth"))


def accuracy(pred, label):
    _, pred = torch.max(pred, dim=1)
    return torch.tensor(torch.sum(pred == label).item() / len(label))


def transformer(image_byte):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    img = Image.open(io.BytesIO(image_byte))
    return transform(img).unsqueeze(0)


def predicts(img):
    pred = model(img)
    _, pred = torch.max(pred, dim=1)
    return pred

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.ImageNet(
    root="data",
    train=True,
    transform=ToTensor()
)

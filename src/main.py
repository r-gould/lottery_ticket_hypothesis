import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Model
from dataset import MNIST
from sparser import Sparser

def main(data_path):
    transform = [
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.5), (0.5)),
    ]

    train_ds = MNIST(data_path, "train", transform)
    test_ds = MNIST(data_path, "test", transform)

    train_dl = DataLoader(train_ds, batch_size=64)
    test_dl = DataLoader(test_ds, batch_size=64)

    model = Model()
    sparser = Sparser(prune_rates={
        "conv" : 0.2,
        "fc" : 0.1,
    }, num_prunes=5)

    learning_rate = 1e-3
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.net.parameters(), lr=learning_rate)
    epochs = 60

    model.compile(loss_fn, optimizer, sparser)
    model.sparse_fit(train_dl, epochs, test_dl)
    model.evaluate(test_dl)

if __name__ == "__main__":
    main("PyTorch/Lottery Ticket Hypothesis/data")
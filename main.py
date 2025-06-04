import random

import torch
import torch.nn.functional as F
from secprint import SectionPrinter as Spt
from torch.utils.data import Subset
from torchjd import backward
from torchjd.aggregation import Mean, UPGrad
from torchvision import transforms
from torchvision.datasets import MNIST

from few_shot_learning_torchjd.criterion import BCELoss, BinaryRegularizationLoss
from few_shot_learning_torchjd.model import MNISTClassifier

Spt.set_automatic_skip(True)


def main():

    # For reproducibility
    random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    N_EPOCHS = 512
    BATCH_SIZE = 20
    USE_JD = False
    LEARNING_RATE = 0.04

    model = MNISTClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # criterion = BinaryRegularizationLoss(
    #     l1_weight=0.0,
    #     l2_weight=0.0,
    #     cross_entropy_weight=0.0,
    #     binary_reg_weight=0.0
    # )
    criterion = BCELoss(
        weight=1.0)
    aggregator = UPGrad() if USE_JD else Mean()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    validation_dataset = MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=Subset(train_dataset, list(range(100))),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=len(validation_dataset),
        shuffle=False,
        num_workers=4,
    )

    #for epoch in range(N_EPOCHS):
    epoch = 0
    while True:
        model.train()
        total_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            data = (data >= 0).float() * 2 - 1
            losses = criterion(model, data, target)
            backward(tensors=losses,
                     aggregator=aggregator)
            optimizer.step()

            total_train_loss += losses.mean().item()

            if batch_idx % 100 == 0:
                Spt.print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataset)}] Loss: {losses.mean().item():.6f}")

        model.eval()
        with torch.no_grad():
            validation_loss = 0
            correct = 0
            for data, target in validation_dataloader:

                data = (data >= 0).float() * 2 - 1
                losses = criterion(model, data, target)
                validation_loss += losses.mean().item()
                pred = model(data).argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        validation_loss /= len(validation_dataloader)
        Spt.print(f"Validation set: Average loss: {validation_loss:.4f}, Accuracy: {correct}/{len(validation_dataset)} ({100. * correct / len(validation_dataset):.0f}%)")
        #model.log_params_distribution()

        if total_train_loss < 0.01:
            Spt.print("Early stopping criterion met, stopping training.")
            break

        epoch += 1

    
    model.eval()
    with torch.no_grad():
        validation_loss = 0
        correct = 0
        for data, target in validation_dataloader:

            data = (data >= 0).float() * 2 - 1
            losses = criterion(model, data, target)
            validation_loss += losses.mean().item()
            pred = model(data).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(validation_dataloader)
    Spt.print(f"Final test: Average loss: {validation_loss:.4f}, Accuracy: {correct}/{len(validation_dataset)} ({100. * correct / len(validation_dataset):.0f}%)")

if __name__ == "__main__":
    main()

import torch
from few_shot_learning_torchjd import MNISTClassifier
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Subset
from secprint import SectionPrinter as Spt
import random
from torchjd import backward
from torchjd.aggregation import UPGrad, Mean

Spt.set_automatic_skip(True)


def main():

    # For reproducibility
    random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    N_EPOCHS = 32
    BATCH_SIZE = 10
    USE_JD = False

    model = MNISTClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(reduce=None)
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
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    for epoch in range(N_EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data)
            losses = criterion(output, target)
            backward(tensors=losses,
                     aggregator=aggregator)
            optimizer.step()

            if batch_idx % 100 == 0:
                Spt.print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataset)}] Loss: {losses.mean().item():.6f}")

        model.eval()
        with torch.no_grad():
            validation_loss = 0
            correct = 0
            for data, target in validation_dataloader:
                output = model(data)
                validation_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        validation_loss /= len(validation_dataset)
        Spt.print(f"Validation set: Average loss: {validation_loss:.4f}, Accuracy: {correct}/{len(validation_dataset)} ({100. * correct / len(validation_dataset):.0f}%)")


if __name__ == "__main__":
    main()

import os
import random
import typing as T
from datetime import datetime

import torch
from secprint import SectionPrinter as Spt
from torch.nn.functional import cosine_similarity
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchjd import backward
from torchjd.aggregation import Mean, UPGrad
from torchvision import transforms
from torchvision.datasets import MNIST

from few_shot_learning_torchjd import MNISTClassifier
from few_shot_learning_torchjd.model import JointMNISTClassifier

Spt.set_automatic_skip(True)

def log_cosine_similarity(writer: SummaryWriter, ) -> T.Callable:
    """Logs the cosine similarity between the aggregation and the average gradient."""
    
    def _log_cosine_similarity(_, inputs: tuple[torch.Tensor], aggregation: torch.Tensor) -> None:
        """Logs the cosine similarity between the aggregation and the average gradient."""
        matrix = inputs[0]
        gd_output = matrix.mean(dim=0)
        similarity = cosine_similarity(aggregation, gd_output, dim=0)
        writer.add_scalar("Cosine Similarity", similarity.item())

    return _log_cosine_similarity

def main():

    # For reproducibility
    random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    N_EPOCHS = 256
    BATCH_SIZE = 20
    N_TRAINING_SAMPLES = 20
    USE_JD = [
              False,
              #True,
              ]
    LEARNING_RATE = [
        0.1, 
        #0.02, 
        #0.03, 
        #0.04, 
        #0.05
    ]

    i = 0
    for use_jd in USE_JD:
        for learning_rate in LEARNING_RATE:
            i += 1
            # if i == 1:
            #     continue
            Spt.print(f"Running experiment with use_jd={use_jd} and learning_rate={learning_rate}")
            run_experiment(
                n_epochs=N_EPOCHS,
                batch_size=BATCH_SIZE,
                n_training_samples=N_TRAINING_SAMPLES,
                use_jd=use_jd,
                learning_rate=learning_rate,
            )
            Spt.print("Experiment completed.")


def run_experiment(
        n_epochs: int,
        batch_size: int,
        n_training_samples: int,
        use_jd: bool,
        learning_rate: float,
        ) -> None:
    """Runs the experiment with the given parameters."""

    model = JointMNISTClassifier(init_mode="zeros") #MNISTClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss(reduction="none") #torch.nn.CrossEntropyLoss(reduction="none")
    aggregator = UPGrad() if use_jd else Mean()

    experiment_name = f"JOINT_{'jd' if use_jd else 'classic'}_{n_training_samples}samples_{learning_rate}lr_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir = os.path.join("runs", experiment_name)
    writer = SummaryWriter(log_dir=log_dir)
    aggregator.register_forward_hook(log_cosine_similarity(writer=writer))

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
        dataset=Subset(train_dataset, list(range(n_training_samples))),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    # writer.add_graph(model, next(iter(train_dataloader))[0])

    for epoch in range(n_epochs):

        with Spt(f"Epoch {epoch + 1}/{n_epochs}", color="blue"):

            model.train()
            train_loss = 0
            for batch_idx, (image, label) in enumerate(train_dataloader):
                optimizer.zero_grad()

                ohe_labels = torch.nn.functional.one_hot(label, num_classes=10).float()
                output = model((image, ohe_labels))
                losses = criterion(output, torch.ones_like(output))
                loss = losses.mean()
                train_loss += loss.item()
                backward(tensors=losses,
                        aggregator=aggregator)
                optimizer.step()

                if epoch == 0:
                    writer.add_images('input/images', image, batch_idx)

                if batch_idx % 100 == 0:
                    Spt.print(f"Train Epoch: {epoch} [{batch_idx * len(image)}/{len(train_dataset)}] Loss: {loss.item():.6f}")

            train_loss /= len(train_dataloader)
            writer.add_scalar('Loss/train', train_loss, epoch)

            model.eval()
            validation_loss = 0
            correct = 0
            with torch.no_grad():
                for input in validation_dataloader:
                    image, target = input
                    prediction = model.predict(image, n_iterations=1000, learning_rate=0.01)

                    correct += prediction.eq(target.view_as(prediction)).sum().item()

            validation_loss /= len(validation_dataloader)
            accuracy = 100. * correct / len(validation_dataset)
            
            writer.add_scalar('Loss/validation', validation_loss, epoch)
            #writer.add_scalar('Accuracy/validation', accuracy, epoch)
            
            Spt.print(f"Validation set: Average loss: {validation_loss:.4f}, Accuracy: {correct}/{len(validation_dataset)} ({accuracy:.0f}%)")

    writer.close()
    Spt.print(f"TensorBoard logs saved to {log_dir}")
    Spt.print("To view TensorBoard, run: tensorboard --logdir=runs")


if __name__ == "__main__":
    main()

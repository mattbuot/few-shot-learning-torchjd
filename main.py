import os
import random
import typing as T
from datetime import datetime

import torch
from prettytable import PrettyTable
from secprint import SectionPrinter as Spt
from torch.nn.functional import cosine_similarity
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchjd import backward
from torchjd.aggregation import Mean, UPGrad
from torchvision import transforms
from torchvision.datasets import MNIST

from few_shot_learning_torchjd import MNISTClassifier
from few_shot_learning_torchjd.criterion import RegularizedLoss
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
              #False,
              True,
              ]
    LEARNING_RATE = [
        0.1, 
        #0.02, 
        #0.03, 
        #0.04, 
        #0.05
    ]
    MSE_WEIGHT, L1_WEIGHT, L2_WEIGHT, KLD_WEIGHT = (1, 0, 0, 0)

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
                l1_weight=L1_WEIGHT,
                l2_weight=L2_WEIGHT,
                mse_weight=MSE_WEIGHT,
                kld_weight=KLD_WEIGHT,
            )
            Spt.print("Experiment completed.")


def run_experiment(
        n_epochs: int,
        batch_size: int,
        n_training_samples: int,
        use_jd: bool,
        learning_rate: float,
        l1_weight: float,
        l2_weight: float,
        mse_weight: float,
        kld_weight: float,
        ) -> None:
    """Runs the experiment with the given parameters."""

    model = JointMNISTClassifier(init_mode=None)#"zeros") #MNISTClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = RegularizedLoss(l1_weight=l1_weight,
                                l2_weight=l2_weight,
                                mse_weight=mse_weight,
                                kld_weight=kld_weight,
                                n_counterfactuals=9,
                                ) #torch.nn.CrossEntropyLoss(reduction="none")
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

    train_dataset_subset = Subset(train_dataset, list(range(n_training_samples)))
    validation_dataset_subset = Subset(validation_dataset, list(range(n_training_samples)))

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    # writer.add_graph(model, next(iter(train_dataloader))[0])

    for epoch in range(n_epochs):

        with Spt(f"Epoch {epoch + 1}/{n_epochs}", color="blue"):

            model.train()
            train_loss = 0
            train_mse_loss = 0
            train_l1_loss = 0
            train_kld_loss = 0
            for batch_idx, (image, label) in enumerate(train_dataloader):
                optimizer.zero_grad()

                ohe_labels = torch.nn.functional.one_hot(label, num_classes=10).float()
                losses, mse_loss, l1_loss, kld_loss = criterion(model=model, model_input=(image, ohe_labels))
                loss = losses.mean()
                train_loss += loss.item()
                train_mse_loss += mse_loss.mean().item()
                train_l1_loss += l1_loss.mean().item()
                train_kld_loss += kld_loss.mean().item()
                backward(tensors=torch.cat([mse_loss, l1_loss.reshape((1, 1)) * criterion.l1_weight]),
                        aggregator=aggregator)
                optimizer.step()

                if epoch == 0:
                    writer.add_images('input/images', image, batch_idx)

                if batch_idx % 100 == 0:
                    Spt.print(f"Train Epoch: {epoch} [{batch_idx * len(image)}/{len(train_dataset)}] Loss: {loss.item():.4f} MSE Loss: {mse_loss.mean().item():.4f} L1 Loss: {l1_loss.mean().item():.4f} KLD Loss: {kld_loss.mean().item():.4f}")

            train_loss /= len(train_dataloader)
            train_mse_loss /= len(train_dataloader)
            train_l1_loss /= len(train_dataloader)
            train_kld_loss /= len(train_dataloader)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('MSELoss/train', train_mse_loss, epoch)
            writer.add_scalar('L1Loss/train', train_l1_loss, epoch)
            writer.add_scalar('KLDLoss/train', train_kld_loss, epoch)

            if True:#epoch % 10 == 5:
                model.eval()

                Spt.print("Evaluating on training set...")
                evaluate_model(model, train_dataloader)

                Spt.print("Evaluating on validation set...")
                evaluate_model(model, validation_dataloader)

                validation_mse_loss = 0
                # with torch.no_grad():
                for input in validation_dataloader:

                    image, target = input
                    ohe_labels = torch.nn.functional.one_hot(target, num_classes=10).float()
                    validation_mse_loss += criterion(model=model, model_input=(image, ohe_labels))[1].mean().item()


                validation_mse_loss /= len(validation_dataloader)
                
                writer.add_scalar('MSELoss/validation', validation_mse_loss, epoch)

                Spt.print(f"Validation set: Average MSE loss: {validation_mse_loss:.4f}")

    writer.close()
    Spt.print(f"TensorBoard logs saved to {log_dir}")
    Spt.print("To view TensorBoard, run: tensorboard --logdir=runs")

def evaluate_model(model: JointMNISTClassifier, data: torch.utils.data.DataLoader) -> None:
    correct = 0

    for input in data:
        image, target = input

        prediction = model.predict_exhaustive(image)
        print_tensor_comparison(prediction, target)

        prediction2 = model.predict(image, n_iterations=100, learning_rate=0.01)
                    #Spt.print(f"Prediction: {prediction}, Target: {target}")
        print_tensor_comparison(prediction2, target)

        correct += prediction.eq(target.view_as(prediction)).sum().item()

        ohe_labels = torch.nn.functional.one_hot(target, num_classes=10).float()
        output, mu, logvar = model((image, ohe_labels))
        valid_output_mean = output.mean().item()
        valid_mu_mean = mu.mean().item()
        valid_logvar_mean = logvar.mean().item()

        wrong_ohe_labels = torch.nn.functional.one_hot((target + 1) % 10, num_classes=10).float()
        wrong_output, mu, logvar = model((image, wrong_ohe_labels))
        wrong_output_mean = wrong_output.mean().item()
        wrong_mu_mean = mu.mean().item()
        wrong_logvar_mean = logvar.mean().item()

        table = PrettyTable(field_names=["Metric", "Valid", "Wrong"])
        table.add_row(["Output Mean", f"{valid_output_mean:.4f}", f"{wrong_output_mean:.4f}"])
        table.add_row(["Mu Mean", f"{valid_mu_mean:.4f}", f"{wrong_mu_mean:.4f}"])
        table.add_row(["LogVar Mean", f"{valid_logvar_mean:.4f}", f"{wrong_logvar_mean:.4f}"])
        Spt.print(str(table))


def print_tensor_comparison(tensor1: torch.Tensor, tensor2: torch.Tensor) -> None:
    """Prints tensor1 with green digits where values match tensor2, red otherwise."""
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")
    
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    
    result = ""
    correct = 0
    for i, (val1, val2) in enumerate(zip(flat1, flat2)):
        if i > 0 and i % tensor1.shape[-1] == 0:
            result += "\n"
        
        color = "\033[92m" if torch.equal(val1, val2) else "\033[91m"
        correct += int(torch.equal(val1, val2))
        reset = "\033[0m"
        result += f"{color}{val1.item():4.0f}{reset} "
    
    result += f"\t{correct}/{flat1.shape[0]}"
    print(result)

if __name__ == "__main__":
    main()

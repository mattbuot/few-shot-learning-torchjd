import torch
import torch.nn.functional as F
from scipy import optimize
from torch import nn


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class JointMNISTClassifier(nn.Module):
    def __init__(self, init_mode: str | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(64 * 7 * 7 + 32, 64)
        self.fc3 = nn.Linear(64, 1)

        if init_mode == "zeros":
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc3.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)
            nn.init.zeros_(self.fc3.bias)
            nn.init.zeros_(self.conv1.weight)
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv1.bias)
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        image, label = x
        
        label = F.relu(self.fc1(label))

        image = F.relu(self.conv1(image))
        image = F.max_pool2d(image, kernel_size=2)
        image = F.relu(self.conv2(image))
        image = F.max_pool2d(image, kernel_size=2)
        image = image.view(-1, 64 * 7 * 7)

        joint_embedding = torch.cat((image, label), dim=1)
        joint_embedding = F.relu(self.fc2(joint_embedding))
        joint_embedding = F.sigmoid(self.fc3(joint_embedding) - 1)
        return joint_embedding
    

    def predict(self, image, n_iterations: int = 1000, learning_rate: float = 0.01) -> torch.Tensor:
        """Given an image, finds the most likely label using gradient descent."""

        labels_for_grad = torch.zeros((image.shape[0], 10), device=image.device, requires_grad=True)

        self.eval()

        optimizer = torch.optim.Adam([labels_for_grad], lr=learning_rate)

        for _ in range(n_iterations):
            optimizer.zero_grad()
            output = self((image, labels_for_grad))
            loss = F.mse_loss(output, torch.ones_like(output))
            loss.backward()
            optimizer.step()

        final_labels = labels_for_grad.detach()
        prediction = torch.argmax(final_labels, dim=1).round().type(torch.int64)
        return prediction

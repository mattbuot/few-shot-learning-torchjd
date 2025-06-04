import torch
import torch.nn.functional as F
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
    
    
    def log_params_distribution(self):
        """Log the distribution of model parameters."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                histogram = param.data.histogram(bins=torch.Tensor([-1.1, -0.8, -0.1, 0.1, 0.8, 1.1]))[0]
                histogram_str = '(' + ', '.join(f'{h.item():.1f}' for h in histogram) + ')'
                print(f"Param: {name}, Shape: {param.data.shape}, Abs Mean: {param.data.abs().mean().item():.2f}, Min: {param.data.min().item():.3f}, Max: {param.data.max().item():.3f}, Histogram: {histogram_str}")

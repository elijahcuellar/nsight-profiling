import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str, device: str = "cpu") -> nn.Module:
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

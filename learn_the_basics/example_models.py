import torch


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=3)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(3 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

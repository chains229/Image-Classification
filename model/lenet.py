from typing import List, Dict, Optional
import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, config):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(config['input_channel'], 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, config['num_classes'])  # Output layer with 10 units for 10 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)  # Flatten the tensor
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)  # No activation as it will be handled by the loss function (e.g., CrossEntropyLoss)
        return x

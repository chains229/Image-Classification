from typing import List, Dict, Optional
import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, config):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(config['input_channel'], 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avg1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, config['num_classes']) 

    def forward(self, x):
        x = self.avg1(self.relu1(self.conv1(x)))
        x = self.avg2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)  
        #x = torch.softmax(x, dim=-1)
        return x

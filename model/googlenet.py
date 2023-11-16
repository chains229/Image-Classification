from typing import List, Dict, Optional
import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Conv2d(in_planes, n1x1, kernel_size=1)

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2)
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1)
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, config):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(config['input_channel'], 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.inception = Inception(192, 64, 96, 128, 16, 32, 32)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.relu3 = nn.ReLU(True)
        self.relu4 = nn.ReLU(True)
        self.relu5 = nn.ReLU(True)
        self.relu6 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avg = nn.AvgPool2d(kernel_size=(1, 1))
        self.fc = nn.Linear(1024, config['num_classes'])
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.maxpool2(x)
        
        x = self.inception(x)
        x = self.inception(x)
        
        x = self.maxpool3(self.relu4(x))
        
        x = self.inception(x)
        x = self.inception(x)
        x = self.inception(x)
        x = self.inception(x)
        x = self.inception(x)
        
        x = self.maxpool3(self.relu5(x))  
        
        x = self.inception(x)
        x = self.inception(x)
        
        x = self.avg(self.relu6(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

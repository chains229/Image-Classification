from transformers import ResNetForImageClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet_50(nn.Module):
    def __init__(self, config):
        super(ResNet_50, self).__init__()
        self.num_classes = config['num_classes']
        self.resnet_50 = models.resnet50(pretrained=True)
        for param in self.resnet_50.parameters():
            param.requires_grad=False
        self.resnet_50.fc = nn.Linear(self.resnet_50.fc.in_features, self.num_classes)
        
    def forward(self, x):
        x = self.resnet_50(x)
        x = F.softmax(x, dim=-1)
        return x

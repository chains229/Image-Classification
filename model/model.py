from typing import List, Dict, Optional
import torch
import torch.nn as nn

from model.lenet import LeNet
from model.googlenet import GoogLeNet
from model.resnet18 import ResNet_18
#from resnet50 import ResNet_50

class CNN_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['model'] == 'lenet':
            self.cnn = LeNet(config)
        elif config['model'] == 'googlenet':
            self.cnn = GoogLeNet(config)
        elif config['model'] == 'resnet18':
            self.cnn = ResNet_18(config)
        else:
            self.cnn = ResNet_50(config)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, imgs, labels=None):
        if labels is not None:
            logits = self.cnn(imgs)
            loss = self.loss_fn(logits, labels)
            return logits, loss
        else:
            logits = self.mlp(imgs)
            return logits

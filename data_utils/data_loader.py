import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms

# MNIST
mnist_dataset_train = datasets.MNIST(root='./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

mnist_dataset_test = datasets.MNIST(root='./data', train=False, download=True,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    
# CIFAR10
cifar_10_dataset_train = datasets.CIFAR10(
                root='./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))

cifar_10_dataset_test = datasets.CIFAR10(
                root='./data', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))
    
# PASCAL VOC 2007 (chưa xong)
voc2007_mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
voc2007_std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
voc2007_transformations = transforms.Compose([transforms.Resize((300, 300)),
                                          transforms.RandomChoice([
                                              transforms.ColorJitter(brightness=(0.80, 1.20)),
                                              transforms.RandomGrayscale(p=0.25)
                                          ]),
                                          transforms.RandomHorizontalFlip(p=0.25),
                                          transforms.RandomRotation(25),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=voc2007_mean, std=voc2007_std),
                                          ])
voc2007_transformations_test = transforms.Compose([transforms.Resize(330),
                                               transforms.FiveCrop(300),
                                               transforms.Lambda(lambda crops: torch.stack(
                                                   [transforms.ToTensor()(crop) for crop in crops])),
                                               transforms.Lambda(lambda crops: torch.stack(
                                                   [transforms.Normalize(mean=voc2007_mean, std=voc2007_std)(crop) for crop in crops])),
                                               ])
voc2007_dataset_train = datasets.VOCDetection(root='./data', year="2007", image_set="train", transform=voc2007_transformations,
                                     download=True)
voc2007_dataset_test = datasets.VOCDetection(root='./data', year="2007", image_set="test", transform=voc2007_transformations_test,
                                     download=True)



'''
voc2007_dataset_train = datasets.VOCDetection(root='./data', year="2007", download=True, image_set="train", 
                                    transform=transforms.Compose([
                                              transforms.Resize((30,50)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ]))
voc2007_dataset_test = datasets.VOCDetection(root='./data', year="2007", download=True, image_set="test", 
                                transform=transforms.Compose([
                                          transforms.Resize((30,50)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ]))
'''

class getDataloader():
    def __init__(self, config):
        self.config = config
        if config['dataset'] == 'mnist':
            self.train_dataset = mnist_dataset_train
            self.test_dataset = mnist_dataset_test
        elif config['dataset'] == 'cifar':
            self.train_dataset = cifar_10_dataset_train
            self.test_dataset = cifar_10_dataset_test
        else:   
            self.train_dataset = voc2007_dataset_train
            self.test_dataset = voc2007_dataset_test

    def get_train(self):
        return DataLoader(self.train_dataset, batch_size = self.config['batch_size'], shuffle = self.config['shuffle'])

    def get_test(self):
        return DataLoader(self.test_dataset, batch_size = self.config['batch_size'], shuffle = self.config['shuffle'])

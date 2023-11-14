import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import idx2numpy
from torchvision import datasets
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# MNIST
mnist_dataset_train = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_train)

mnist_dataset_test = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_test)
    
# CIFAR10
cifar_10_dataset_train = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

cifar_10_dataset_test = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
    
# PASCAL VOC 2007 (chưa xong)
'''
class PASCAL_VOC_2007_Dataset():
    def __init__(self):
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
'''        
    
class getDataloader():
    def __init__(self, config):
        if config['dataset'] == 'mnist':
            self.train_dataset = mnist_dataset_train
            self.test_dataset = mnist_dataset_test
        
        elif config['dataset'] == 'cifar':
            self.train_dataset = cifar_10_dataset_train
            self.test_dataset = cifar_10_dataset_test
        
        '''
        else:   
            self.train_dataset = PASCAL_VOC_2007_Dataset()
            self.test_dataset = test_images_file
        '''

    def get_train(self):
        return DataLoader(self.train_dataset, batch_size = self.config['batch_size'], shuffle = self.config['shuffle'])

    def get_test(self):
        return DataLoader(self.test_dataset, batch_size = self.config['batch_size'], shuffle = self.config['shuffle'])

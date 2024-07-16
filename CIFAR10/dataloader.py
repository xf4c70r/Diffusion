# Import Libraries
import torch
import torchvision
from torchvision import transforms 

import warnings
warnings.filterwarnings('ignore')

from main import IMG_SIZE, BATCH_SIZE

# Dataloader 

class Dataloader:
    def __init__(self):
        self.transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]

    def load_transformed_dataset(self):
        data_transform = transforms.Compose(self.transforms)
        train = torchvision.datasets.CIFAR10(root='.', train=True, transform=data_transform, 
                                    download=True)
        test = torchvision.datasets.CIFAR10(root='.', train=False, transform=data_transform, 
                                    download=True)
        return torch.utils.data.ConcatDataset([train, test])
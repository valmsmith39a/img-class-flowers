import torch
import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image

from utils import ImgClassifierUtils
from model import ModelLucy

# Number of input units for classifier must match the chosen pre-trained model architecture attributes specifications 
# Example 1 (no architecture specified, defaults to vgg19): python3 train.py flowers
# Example 2 (Densenet): python3 train.py flowers --save_dir checkpoint.pth --arch densenet121 --input_units 1024 --hidden_units 512  --epochs 1 --learning_rate .001 --gpu
parser = argparse.ArgumentParser(description='Classify images of flowers - Training')
parser.add_argument('data_dir', action="store", type=str, help="Please enter data directory path")
parser.add_argument('--save_dir', action="store", dest="save_dir", type=str, help="Please enter save model checkpoint directory path")
parser.add_argument('--arch', action="store", dest="arch_type", type=str, help="Please enter architecture type")
parser.add_argument('--input_units', action="store", dest="input_units", type=int, help="Please enter number of input units")
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, help="Please enter number of hidden units")
parser.add_argument('--epochs', action="store", dest="num_of_epochs", type=int, help="Please enter number of epochs")
parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, help="Please enter architecture type")
parser.add_argument('--gpu', action="store_true", default=False)
args = parser.parse_args()

def train_model():
    data_dir = args.data_dir or 'flowers'
    save_dir = args.save_dir or 'checkpoint.pth'
    device_type = args.gpu
    
    if device_type:
        device = 'cuda'
    else:
        device = ModelLucy.get_device() # default is available device

    # Set parameters
    num_of_epochs = args.num_of_epochs or 1
    learning_rate = args.learning_rate or .001
    input_units = args.input_units or 25088
    hidden_units = args.hidden_units or 4096
    arch_type = args.arch_type or 'vgg19'

    options = {
        'device': device,
        'input_units': input_units,
        'hidden_units': hidden_units,
        'arch_type': arch_type
    }

    image_datasets, dataloaders = ImgClassifierUtils.load_data(data_dir)

    model = ModelLucy.build_network(options)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    model = ModelLucy.train_network(model, criterion, optimizer, dataloaders, num_of_epochs, device)
    
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint_data = {
        'input_units': input_units,
        'hidden_layers': [hidden_units],
        'class_to_idx': model.class_to_idx,
        'learning_rate': learning_rate,
        'arch_type': arch_type
    }

    ModelLucy.save_checkpoint(save_dir, model, checkpoint_data)

train_model()

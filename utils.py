import torch
import json
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

class ImgClassifierUtils:
    
    def cat_to_name():
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f) 
        return cat_to_name

    def load_data(data_dir_path):
        data_dir = data_dir_path or 'flowers'
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        # Define transforms for training, validation and testing data
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(45),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }

        dirs = {
            'train': train_dir,
            'valid': valid_dir,
            'test': test_dir
        }

        image_datasets = {
            x: datasets.ImageFolder(dirs[x], transform=data_transforms[x]) for x in ['train', 'valid', 'test']
        }

        dataloaders ={
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid', 'test']}

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

        classnames = image_datasets['train'].classes
        
        return [image_datasets, dataloaders]
    
    def process_image(image_path):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

        img = Image.open(image_path)

        # Constrain the smaller of width and height to 256
        if img.size[0] > img.size[1]:
            img.thumbnail((1000000, 256))
        else:
            img.thumbnail((256, 2000000))

        # Crop the center of the image

        # Set margins
        left_margin = (img.width - 224) / 2
        right_margin = left_margin + 224
        bottom_margin = (img.height - 224) / 2
        top_margin = bottom_margin + 224
        img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
        img = np.array(img) / 255

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = img.transpose((2, 0, 1))

        return img
    
    
        
     
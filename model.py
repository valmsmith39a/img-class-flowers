import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict

from utils import ImgClassifierUtils

class ModelLucy:
            
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def build_network(options):
        # Build network
        
        device = options['device']
        input_units = options['input_units'] or 25088
        hidden_units = options['hidden_units'] or 4096
        arch_type = options['arch_type']
    
        # Select pre-trained model for transfer learning features
        if arch_type == 'vgg19':
            model = models.vgg19(pretrained=True)
            print('pre-trained model is vgg19', model)
        elif arch_type == 'densenet121':
            model = models.densenet121(pretrained=True)
            print('pre-trained model is densenet121', model)
        else:
            # Default to vgg19
            model = models.vgg19(pretrained=True)
            print('pre-trained model is vgg 19', model)

        # Turn off gradients for model
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_units, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        model.classifier = classifier

        model = model.to(device)
        
        return model
        
    def train_network(model, criterion, optimizer, dataloaders, num_epochs=10, device='cuda'):
        steps = 0
        running_loss = 0
        print_every = 5

        for epoch in range(num_epochs):
            for images, labels in dataloaders['train']:
                steps += 1
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Validation
                if steps % print_every == 0:
                    validation_loss = 0
                    accuracy = 0
                    model.eval()

                    with torch.no_grad():
                        for inputs, labels in dataloaders['valid']:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            validation_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(
                        f"Epoch {epoch+1}/{num_epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
                        f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}"
                    )
                    running_loss = 0
                    model.train()
        print('Training complete')
        return model

    def predict(model, image_path, options):
        ''' Predict the class (or classes) of an image using a trained deep learning model.'''

        img = ImgClassifierUtils.process_image(image_path)
        top_k = options['top_k'] or 5
        image_tensor = torch.from_numpy(img).type(torch.FloatTensor)

        model_input = image_tensor.unsqueeze(0)
        ps = torch.exp(model(model_input))
        top_p, top_class = ps.topk(top_k, dim=1)
        
        return top_p, top_class
    
    def save_checkpoint(filepath, model, checkpoint_data):
        class_to_idx = checkpoint_data['class_to_idx'] or {}
        input_units = checkpoint_data['input_units'] or 25088
        hidden_layers = checkpoint_data['hidden_layers'] or [4096]
        learning_rate = checkpoint_data['learning_rate'] or .001
        arch_type = checkpoint_data['arch_type'] or 'vgg19'

        checkpoint = {'input_size': input_units,
              'output_size': 102,
              'hidden_layers': hidden_layers,
              'state_dict': model.state_dict(),
              'class_to_idx': class_to_idx,
              'learning_rate': learning_rate,
              'dropout': 0.2,
              'arch_type': arch_type,
              'batch_size': 64}
        
        checkpoint_save_path = filepath or 'checkpoint.pth'
        torch.save(checkpoint, checkpoint_save_path)

    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
    
        input_size = checkpoint['input_size']
        output_size = checkpoint['output_size']
        hidden1 = checkpoint['hidden_layers'][0]
        dropout = checkpoint['dropout']
        state_dict = checkpoint['state_dict']
        class_to_idx = checkpoint['class_to_idx']
        arch_type = checkpoint['arch_type']
    
        model = models.vgg19(pretrained=True)
        
        if arch_type == 'vgg19':
            model = models.vgg19(pretrained=True)
        elif arch_type == 'densenet121':
            model = models.densenet121(pretrained=True)
        else:
            # Default to vgg19
            model = models.vgg19(pretrained=True)

        # Turn off gradients for model
        for param in model.parameters():
            param.requires_grad = False
    
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden1)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(dropout)),
            ('fc2', nn.Linear(hidden1, output_size)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    
        model.classifier = classifier

        model.load_state_dict(state_dict)

        model.class_to_idx = class_to_idx
    
        return model
   
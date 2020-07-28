# coding: ascii
import os
import time
import copy
import argparse
import json
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from flowerclassifier import create_model

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            dataset_size = 0
            for inputs, labels in dataloaders[phase]:
                batch_size = inputs.size()[0]
                dataset_size = dataset_size + batch_size
                print( '|' * batch_size, end='',  flush=True)
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            print()
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
        
def main():
    # Define accepted arguments
    parser = argparse.ArgumentParser(description='Train flower classifier')
    parser.add_argument('data_dir', help='directory path where training and validation datasets are stored')
    parser.add_argument('--save_dir', type=str, action='store', help='directory path where to store checkpoint')
    parser.add_argument('--gpu', action='store_true', help='enable gpu usage')
    parser.add_argument('--learning_rate', type=float, default=0.001, action='store', help='set learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, action='store', help='set number of hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=20, action='store', help='set number of epochs')
    parser.add_argument('--arch', type=str, choices=['vgg16', 'resnet18'], default='vgg16', action='store', help='set pretrained model architecture')
    
    
    # Parse arguments
    args = parser.parse_args()
    checkpoint_path = os.path.join(args.save_dir, 'model.pkl') if args.save_dir is not None else None
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    hidden_units = args.hidden_units
    device = "cuda" if args.gpu else "cpu"
    epochs = args.epochs
    learning_rate = args.learning_rate
    arch = args.arch
    
    # Define your transforms for the training, validation, and testing sets
    data_transforms = { 'train': transforms.Compose([
                                                     transforms.RandomResizedCrop(224),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                    ]),
                       'valid': transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                    ]) }

    # Load the datasets with ImageFolder
    image_datasets = {'train': ImageFolder(train_dir, data_transforms['train']),
                      'valid': ImageFolder(valid_dir, data_transforms['valid']) }
                      

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = { 'train': DataLoader(image_datasets['train'], batch_size=8,shuffle=True, num_workers=2),
                    'valid': DataLoader(image_datasets['valid'], batch_size=8,shuffle=True, num_workers=2)}
    
    # Instanciate model accroding to arch and hidden units
    model = create_model(arch, hidden_units)
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    # Move it to the requested device, gpu or cpu
    model = model.to(device)

    # Choose cross entropy loss as it is a category optimization problem
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)

    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, num_epochs=epochs, device=device)

    # Save the checkpoint
    if checkpoint_path is not None:
        model.to('cpu')
        torch.save({
                    'class_to_idx': model.class_to_idx,
                    'classifier_state_dict': model.classifier.state_dict(),
                    'hidden_units' :  model.classifier.hidden_units,
                    'arch' :  arch
                    }, checkpoint_path) 
    return

if __name__ == "__main__":
    # Execute only if run as a script
    main()

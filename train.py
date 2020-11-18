# Imports here
from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
import json
import time
import os
import matplotlib.pyplot as plt
import argparse

def get_user_args():
  
    parser = argparse.ArgumentParser(description='Train Image Classifer')

    parser.add_argument ('data_dir', type = str, help = 'Data directory where flower images are located')
    parser.add_argument ('--save_dir', type = str, default=os.getcwd(), help = 'Directory where checkpoint will be saved. (Default: current working directory)')
    parser.add_argument ('--arch', type = str, default='densenet121',
                         help = "Select architecture from torchvision.models.  Choose one: 'densenet121', 'resnet50', 'vgg16'. (Default='densenet121')")
    parser.add_argument ('--learning_rate',  type = float, default=0.002, help = 'Learning rate, (default: 0.002)')
    parser.add_argument ('--hidden_units',type = int, default=512, help = 'Hidden units in the classifier, (default: 512)')
    parser.add_argument ('--epochs', type = int, default=3, help = 'Number of epochs, (default: 3)')
    parser.add_argument ('--gpu', action='store_true', default=True, help = 'Enable GPU (default: enabled)')

    return parser.parse_args()


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return train_data, valid_data, test_data, trainloader, validloader, testloader

    
def assign_processor(gpu_arg):
    
    if not gpu_arg:
        print('CPU selected')
        return torch.device('cpu')    
  
    else:  # Use GPU if it's available
        if torch.cuda.is_available():
            print('GPU selected')
            return torch.device('cuda')
        else:
            print('GPU selected but not available, using CPU instead')
            return torch.device('cpu')

def build_model(device, arch='densenet121', hidden_units=512, p_dropout=0.2, learn_rate=0.002):
    
    model = eval('models.{}(pretrained=True)'.format(arch))
    use_resnet = False
    if arch == 'resnet50':  
        use_resnet = True
        in_features = model.fc.in_features
    
    elif arch == 'densenet121':
        in_features = model.classifier.in_features
        
    elif arch == 'vgg16':
        in_features = model.classifier[0].in_features
    
    
    for param in model.parameters():
        param.requires_grad = False
    
    
    # Define classifier for flower images
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('dropout1', nn.Dropout(p_dropout)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    if use_resnet:
        # if resnet architecture is used then it uses fc layer instead of a classifier at output
        model.fc = classifier
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)
    else:
        model.classifier = classifier
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)    
    
    criterion = nn.NLLLoss()
    
    model.to(device)
    
    return model, criterion, optimizer


def train_model(model, trainloader, validloader, device, optimizer, criterion, epochs=3, print_every=5):
    since = time.time()
    
    steps = 0
    running_loss = 0
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        valid_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def save_checkpoint(save_dir, model, arch, optimizer, train_data):
   
    model.class_to_idx = train_data.class_to_idx
    
    if arch == 'resnet50':
        checkpoint = {
              'classifier': model.fc,
              'opt_state': optimizer.state_dict,
              'class_to_idx': model.class_to_idx,
              'arch': arch,
              'model_state_dict': model.state_dict()}
    else:
        checkpoint = {
              'classifier': model.classifier,
              'opt_state': optimizer.state_dict,
              'class_to_idx': model.class_to_idx,
              'arch': arch,
              'model_state_dict': model.state_dict()}
    
    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    print('Checkpont saved at: ', save_dir)  
      
 
def main():
    # Process user inputs
    args = get_user_args()
    if args.arch not in ['densenet121','resnet50','vgg16']:
        print("\nWrong model selected.  Please select from one of the following: 'densenet121', 'resnet50', 'vgg16'\n")  
        exit()

    # Load image data for data directory to train network
    train_data, valid_data, test_data, trainloader, validloader, testloader = load_data(args.data_dir)
    
    # Set device on with either CPU or GPU based on user input
    device = assign_processor(args.gpu)
    
    # Build model with user selected network architecture user specified hyperparameters (learning rate, number of hidden units)
    model, criterion, optimizer = build_model(device, args.arch, args.hidden_units, args.learning_rate)
    
    # Train the model with user specified number of epochs
    train_model(model, trainloader, validloader, device, optimizer, criterion, args.epochs)
    
    # Save checkpoint as checkpoint
    save_checkpoint(args.save_dir, model, args.arch, optimizer, train_data)
    
# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()
    
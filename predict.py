# Imports here
from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
import json
import os
#import matplotlib.pyplot as plt
import argparse
from prettytable import PrettyTable

def get_user_args():
    parser = argparse.ArgumentParser(description='Predict class for input image')
    parser.add_argument('input', type=str, help="Complete path of the image, for example: 'flowers/test/37/image_03734.jpg'")
    parser.add_argument('checkpoint', type=str, help="'Filename of saved model checkpoint.  For example: 'checkpoint.pth'")
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes. (Default: 5)')
    parser.add_argument('--category_names', type=str, help='Use a mapping of categories to real names from a json file. (Default: cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available. (Default: Enable GPU)')
    
    return parser.parse_args()


        
def load_checkpoint(filepath, gpu):
    # Load saved checkpoint
    #checkpoint = torch.load(filepath)
    if gpu==False:
        print('CPU selected')
        checkpoint = torch.load(filepath, map_location='cpu')
    else:
        if torch.cuda.is_available():
            print('GPU selected')
            checkpoint = torch.load(filepath)
        else:
            print('GPU selected but not available, using CPU instead')
            checkpoint = torch.load(filepath, map_location='cpu')
    
    # Download pre-trained model
    arch = checkpoint['arch']
    print('Model architecture loaded:',arch)
    model = eval('models.{}(pretrained=True)'.format(arch))
    
    
    if arch[:6] == 'resnet':  
        model.fc = checkpoint['classifier']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    else:
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
     
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    width, height = im.size
    
   
    # resize image
    if width >= height:
        im.thumbnail((256 * width / height, 256))
    else:
        im.thumbnail((256, 256 * height / width))
           
    crop_sq_size = 224 

    left = (width/2 - crop_sq_size) / 2 
    top = (height/2 - crop_sq_size) / 2
    right = (left + crop_sq_size)
    bottom = (top + crop_sq_size)
    
    im = im.crop((left, top, right, bottom))
    
    np_image = np.array(im) / 255     #to make values from 0 to 1

    means = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]
    
    np_image = (np_image - means) / std_dev
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    cuda = torch.cuda.is_available()
    # TODO: Implement the code to predict the class from an image file
    if cuda:
        model.cuda()
    else:
        model.cpu()
    
    model.eval()
    
    img = process_image(image_path)
    img = torch.from_numpy(img).float()#.cuda()
    img = torch.unsqueeze(img, dim=0)
    
    if cuda:
        img = img.cuda()
    
    output = model.forward(img)
    preds = torch.exp(output).topk(topk)
    probs = preds[0][0].cpu().data.numpy()
    classes = preds[1][0].cpu().data.numpy()
    topk_labels = [idx_to_class[i] for i in classes]
    return probs.tolist(), topk_labels

def cat_to_name(file):
    with open(file, 'r') as f:
        cat_to_name = json.load(f)  
    return cat_to_name

def get_flower_names(classes, flowers):
    labels = []
    for cl in classes:
        labels.append(flowers[cl])
    return labels


def print_results(category_names, prob, classes):
    table = PrettyTable()
    if category_names == None:
        # If category names json file is not provided, then prints out class number and top k probabilities in a tabular format
        column_names = ['Class Number', 'Probability']
        table.add_column(column_names[0], classes)
        table.add_column(column_names[1], prob)
        print(table)
        print('Predicted flower category number: ', classes[0])
    else:        
        # If category names json file is not provided, then prints out flower name and top k probabilities in a tabular format
        flowers_lookup = cat_to_name(category_names)
        labels = get_flower_names(classes, flowers_lookup)
        column_names = ['Flower Name', 'Probability']
        table.add_column(column_names[0], labels)
        table.add_column(column_names[1], prob)
        print(table)
        print('Predicted flower name: ',labels[0])
           
def main():
    
    # Prase user input
    args = get_user_args()
   
    # Load previously saved checkpoint
    model = load_checkpoint(args.checkpoint, args.gpu)
    
    # Generate predictions
    prob, classes = predict(args.input, model, args.top_k) 
    
    # Print results in a table
    print_results(args.category_names, prob, classes)

    
# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()
    
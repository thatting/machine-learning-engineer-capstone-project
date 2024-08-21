####################################################################################################
# Following references have been used in Project Image Classification
# - Starter code provided by Udacity in the projects of the AWS Machine Learning Engineer Nanodegree
# - The official documentation for Amazon SageMaker: https://docs.aws.amazon.com/sagemaker/
####################################################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import copy
import os
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import smdebug.pytorch as smd

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, hook):
    '''
    This function takes a model and test dataloader and provides the test loss and accuracy
    '''
    model.eval()
    running_loss=0
    running_corrects=0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Added 240724: Check if CUDA is available
    if hook:
        hook.set_mode(smd.modes.EVAL) # Set hook to EVAL mode
    for inputs, labels in test_loader:
        inputs = inputs.to(device)  # Added 240724: Offload to the GPU
        labels = labels.to(device)  # Added 240724: Offload to the GPU
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    total_loss = running_loss // len(test_loader)
    total_acc = running_corrects.double() // len(test_loader)
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")


def train(model, train_loader, validation_loader, criterion, optimizer, epochs, hook):
    '''
    This function takes a model and dataloaders and then trains the model
    '''
    epochs=epochs
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Added 240724: Check if CUDA is available
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        if hook:
            hook.set_mode(smd.modes.TRAIN) # Set hook to TRAIN mode
        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in image_dataset[phase]:
                inputs = inputs.to(device)  # Added 240724: Offload to CUDA
                labels = labels.to(device)  # Added 240724: Offload to CUDA
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects // len(image_dataset[phase])
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,
                                                                                 epoch_loss,
                                                                                 epoch_acc,
                                                                                 best_loss))
    return model


def net():
    '''
    This function initializes the pretrained model and adds an extra layer with 5 outputs    
    '''
    model = models.resnet101(pretrained=True)  # Changed to resnet101 
    for param in model.parameters():
        param.requires_grad = False   
    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 5))  # Number of classes is 5
    return model


def create_data_loaders(data, batch_size):
    '''
    This function creates the dataloaders for training, validation and testing
    '''
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    return train_data_loader, test_data_loader, validation_data_loader


def main(args):
    '''
    Initialize a model by calling the net function
    '''
    logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size)
    model=net().to(device)  # Added 240724: Offload to the GPU
    '''
    Create the loss criterion and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(ignore_index=5)   # ignore_index is set 5 corresponding to 5 classes
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    '''
    Call the train function to start model training, including hook for debugging
    '''
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
    logger.info("Starting Model Training")    
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, args.epochs, hook)
    '''
    Test the model to see accuracy
    '''
    logger.info("Testing Model") 
    test(model, test_loader, loss_criterion, hook)
    '''
    Save the trained model
    '''
    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__=='__main__':
    '''
    Specify the hyperparameters needed to train the model
    '''
    parser=argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args=parser.parse_args()
    print(args)
    main(args)

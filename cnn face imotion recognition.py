#!/usr/bin/env python3
# -*- coding: utf-8 -*-
!pip install deepsplines
import torch
import time
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
# Need to import dsnn (takes the role of torch.nn for DeepSplines)
sys.path.append('/kaggle/input/code-resnet-fer2013')
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import random_split
if not os.path.isdir('./ckpt'):
    print(f'\nLog directory ./ckpt not found. Creating it.')
    os.makedirs('./ckpt')

params = {
        'net': 'cnn_relu',
        'device': "cuda:0" if torch.cuda.is_available() else "cpu",
        'log_dir': './ckpt',
        'num_epochs': 100,
        'milestones': [150, 225, 262],
        'activation_type': 'relu',
        'save_memory': False,
        'lipschitz': False,
        'lmbda': 1e-4,
        'lr': 1e-1,
        'aux_lr': 1e-3,
        'weight_decay': 5e-4,
        'log_step': 5,  # 8 times per epoch
        'valid_log_step': -1,  # once every epoch
        'test_as_valid': True,  # print test loss at validation
        'dataset_name': 'fer2013',
        'batch_size': 64,
        'plot_imgs': False,
        'verbose': False,
        'num_classes': 7,
    }

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 128)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.fc1_relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batch_norm1(x)
        x = self.maxpool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.batch_norm2(x)
        x = self.maxpool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.batch_norm3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.batch_norm4(x)
        x = self.fc1_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.softmax(x)

        return x
# Instantiate the model
net = net()
# Print the model summary
print(net)

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # add regularization loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        train_loss = running_loss / len(train_loader)
    return  all_predictions, all_labels, train_loss


def test_model(model, test_loader, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            test_loss = running_loss / len(test_loader)
    return  all_predictions, all_labels, test_loss


def plot_loss_accuracy(train_losses, validation_losses, train_accuracies, validation_accuracies):
    # Plot Training and Validation Loss
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    ########################################################################
    # Load the data

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    batch_size = params['batch_size']

    trainset = torchvision.datasets.FER2013(root='/kaggle/input/fer2013',
                                            split='train',
                                            transform=transform)
    # Define the split ratios
    train_ratio = 0.9
    valid_ratio = 0.1

    # Calculate the sizes of training and validation sets
    train_size = int(train_ratio * len(trainset))
    valid_size = len(trainset) - train_size

    # Split the training set into training and validation sets
    trainset, validset = random_split(trainset, [train_size, valid_size])

    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)

    testset = torchvision.datasets.FER2013(root='/kaggle/input/fer2013',
                                           split='train',
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4)
    
    classes = ( 'Angry','Disgust','Fear','Happy', 'Sad', 'Surprised', 'Neutral')


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'\nDevice: {device}')

    ########################################################################
    # Network, optimizer, loss

    net.to(device)
    print('ReLU: nb. parameters - {:d}'.format(sum(p.numel() for p in net.parameters())))

    # Wrap the model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
        
    #net.load_state_dict(torch.load('/kaggle/input/model-resnet-fer2013-ds/model_resnetferds.pth'))

    optimizer = optim.SGD(net.module.parameters(), lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    ########################################################################
 
    # Lists to store training loss and test accuracy for each epoch
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['milestones'], gamma=0.1)

    ########################################################################

    print('\nTraining relu network.')
    start_time = time.time()

    for epoch in range(params['num_epochs']):
        train_predictions, train_labels,train_loss = train_model(net, trainloader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_accuracies.append(train_accuracy)
        # Learning rate scheduling step
        scheduler.step()
        print(f'Epoch {epoch + 1}/{params["num_epochs"]}')
        print(f'train_acc :{train_accuracy* 100:.2f}%')
        print(f'train loss :{train_loss:.4f}')
        predictions, labels,validation_loss = test_model(net, validloader, device)
        acc_score = accuracy_score(labels, predictions)
        validation_accuracies.append(acc_score)
        validation_losses.append(validation_loss)

        if epoch % params['log_step'] == 0:
            if epoch % params['valid_log_step'] == 0:

                print(f'validation_acc: {acc_score * 100:.2f}%')
                print(f'validation_loss: {validation_loss:.4f}')

                # Confusion Matrix
                cm = confusion_matrix(labels, predictions)
                print("Confusion Matrix:")
                print(cm)

                # Plot confusion matrix for validation
                plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                plt.title("Confusion Matrix on validation set")
                plt.colorbar()
                plt.xticks(range(params['num_classes']), classes, rotation=45)
                plt.yticks(range(params['num_classes']), classes)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.show()
                
    end_time = time.time()

    print('Finished Training relu network. \n'
          'Took {:d} seconds. '.format(int(end_time - start_time)))
    
    #plot train and validation
    plot_loss_accuracy(train_losses,validation_losses, train_accuracies,validation_accuracies)

    ######################################################################

    # Test the model on the test set after training
    test_predictions, test_labels ,test_loss= test_model(net, testloader, device)
    test_acc_score = accuracy_score(test_labels, test_predictions)
    print(f'Final Accuracy on test set: {test_acc_score * 100:.2f}%')
    print(f'Final Test Loss: {test_loss:.4f}')

    # Confusion Matrix for test set
    test_cm = confusion_matrix(test_labels, test_predictions)
    print("Confusion Matrix for Test Set:")
    print(test_cm)

    # Plot confusion matrix for test set
    plt.imshow(test_cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Test Set")
    plt.colorbar()
    plt.xticks(range(params['num_classes']), classes, rotation=45)
    plt.yticks(range(params['num_classes']), classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    

    # Save the trained model
    model_filename = 'model_resnetfer.pth'
    torch.save(net.state_dict(), model_filename)

    #net = net()
    #net = nn.DataParallel(net)
    #net.load_state_dict(torch.load('/kaggle/working/model_resnetfer.pth'))


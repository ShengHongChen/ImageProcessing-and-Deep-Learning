import os
import numpy
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from ResNet50 import ResNet50

import matplotlib.pyplot as plt

from PIL import Image

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define your custom dataset paths
train_path = './training_dataset/'
val_path = './validation_dataset/'

train_imgs = os.listdir(train_path)
val_imgs = os.listdir(val_path)

# Class Distribution
dogs_list = [img for img in train_imgs if img.split(" ")[0] == "Dog"]
cats_list = [img for img in train_imgs if img.split(" ")[0] == "Cat"]

print("No of Dogs Images: ",len(dogs_list))
print("No of Cats Images: ",len(cats_list))

class_to_int = {"Dog" : 0, "Cat" : 1}
int_to_class = {0 : "Dog", 1 : "Cat"}

# Dataset Class - for retriving images and labels
class CatDogDataset(Dataset):
    
    def __init__(self, imgs, class_to_int, mode = "train", transforms = None):
        
        super().__init__()
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms
        
    def __getitem__(self, idx):
        
        image_name = self.imgs[idx]
        
        img = Image.open(train_path + image_name)
        img = img.resize((224, 224))
        
        if self.mode == "train" or self.mode == "val":
        
            # Preparing class label
            label = self.class_to_int[image_name.split(" ")[0]]
            label = torch.tensor(label, dtype = torch.float32)

            # Apply Transforms on image
            img = self.transforms(img)

            return img, label
        
        elif self.mode == "test":
            
            # Apply Transforms on image
            img = self.transforms(img)

            return img
        
    def __len__(self):
        return len(self.imgs)
    
def transform_without_erasing():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels = 3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0),(1, 1, 1)),
    ])

def transform_with_erasing():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels = 3),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0),(1, 1, 1)),
])

train_dataset = CatDogDataset(train_imgs, class_to_int, mode = "train", transforms = transform_without_erasing())
val_dataset = CatDogDataset(val_imgs, class_to_int, mode = "val", transforms = transform_without_erasing())


batch_size = 16
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True)


# Function to calculate accuracy
def accuracy(preds, trues):
    
    # Converting preds to 0 or 1
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    
    # Calculating accuracy by comparing predictions with true labels
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    
    # Summing over all correct predictions
    acc = numpy.sum(acc) / len(preds)
    
    return (acc * 100)

NUM_EPOCHS = 30
model = ResNet50()
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)

criterion = nn.BCELoss()

# Logs - Helpful for plotting after training finishes
train_logs = {"loss" : [], "accuracy" : [], "time" : []}
val_logs = {"loss" : [], "accuracy" : [], "time" : []}

# Function - One Epoch Train
def train_one_epoch(train_data_loader):
    
    # Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    # Iterating over data loader
    for images, labels in train_data_loader:
        
        # Loading images and labels to device
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
        # Reseting Gradients (reset)
        optimizer.zero_grad()
        
        # Forward
        preds = model(images)
        
        # Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        # Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
        
        # Backward (update)
        _loss.backward()
        optimizer.step()
    
    # Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    # Acc and Loss
    epoch_loss = numpy.mean(epoch_loss)
    epoch_acc = numpy.mean(epoch_acc)
    
    # Storing results to logs
    train_logs["loss"].append(epoch_loss)
    train_logs["accuracy"].append(epoch_acc)
    train_logs["time"].append(total_time)
        
    return epoch_loss, epoch_acc, total_time

# Function - One Epoch Valid
def val_one_epoch(val_data_loader, best_val_acc):
    
    # Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    # Iterating over data loader
    for images, labels in val_data_loader:
        
        # Loading images and labels to device
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
        # Forward
        preds = model(images)
        
        # Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        # Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
    
    # Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    # Acc and Loss
    epoch_loss = numpy.mean(epoch_loss)
    epoch_acc = numpy.mean(epoch_acc)
    
    # Storing results to logs
    val_logs["loss"].append(epoch_loss)
    val_logs["accuracy"].append(epoch_acc)
    val_logs["time"].append(total_time)
    
    # Saving best model
    if epoch_acc > best_val_acc:
        best_val_acc = epoch_acc
        torch.save(model.state_dict(), "ResNet50_Model Weights.pth")
        
    return epoch_loss, epoch_acc, total_time, best_val_acc

# Start training and validation without erasing
best_val_acc = 0

for epoch in range(NUM_EPOCHS):
    
    # Training
    loss, acc, _time = train_one_epoch(train_loader)
    
    # Print Epoch Details
    print("\nTraining")
    print("Epoch {}".format(epoch+1))
    print("Loss : {}".format(round(loss, 4)))
    print("Acc : {}".format(round(acc, 4)))
    print("Time : {}".format(round(_time, 4)))
    
    # Validation
    loss, acc, _time, best_val_acc = val_one_epoch(val_loader, best_val_acc)
    
    # Print Epoch Details
    print("\nValidating")
    print("Epoch {}".format(epoch+1))
    print("Loss : {}".format(round(loss, 4)))
    print("Acc : {}".format(round(acc, 4)))
    print("Time : {}".format(round(_time, 4)))

max_accuracy_without_erasing = int(best_val_acc)


train_dataset = CatDogDataset(train_imgs, class_to_int, mode = "train", transforms = transform_with_erasing())
val_dataset = CatDogDataset(val_imgs, class_to_int, mode = "val", transforms = transform_with_erasing())

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True)


train_logs = {"loss" : [], "accuracy" : [], "time" : []}
val_logs = {"loss" : [], "accuracy" : [], "time" : []}

# Start training and validation with erasing
best_val_acc = 0

for epoch in range(NUM_EPOCHS):
    
    # Training
    loss, acc, _time = train_one_epoch(train_loader)
    
    # Print Epoch Details
    print("\nTraining")
    print("Epoch {}".format(epoch+1))
    print("Loss : {}".format(round(loss, 4)))
    print("Acc : {}".format(round(acc, 4)))
    print("Time : {}".format(round(_time, 4)))
    
    # Validation
    loss, acc, _time, best_val_acc = val_one_epoch(val_loader, best_val_acc)
    
    # Print Epoch Details
    print("\nValidating")
    print("Epoch {}".format(epoch+1))
    print("Loss : {}".format(round(loss, 4)))
    print("Acc : {}".format(round(acc, 4)))
    print("Time : {}".format(round(_time, 4)))

max_accuracy_with_erasing = int(best_val_acc)


class_labels = ['Without Random erasing', 'With Random erasing']
accuracy = [max_accuracy_without_erasing, max_accuracy_with_erasing]

# Create a bar chart
bar_width = 0.35
index = numpy.arange(len(class_labels))
bars = plt.bar(index, accuracy, bar_width)

fig = plt.figure()

plt.bar(class_labels, accuracy)
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')

# Display the values on top of the bars
for bar, acc in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.05, bar.get_height() + 1, f'{acc}', ha = 'center')

fig.savefig('Show Comparison.png')

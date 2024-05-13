import time

import torch
import torchvision
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

import matplotlib.pyplot as plt

from VGG19 import VGG19

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    torchvision.transforms.Grayscale(num_output_channels = 3),
    transforms.ToTensor(),
])

batch_size = 64
train_dataset = datasets.MNIST(root = 'data', train = True, transform = transform, download = True)
valid_dataset = datasets.MNIST(root = 'data', train = False, transform = transform)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False)

NUM_EPOCHS = 50
model = VGG19(num_classes = 10)
model = model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

def compute_accuracy_and_loss(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
         
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        cross_entropy += F.cross_entropy(logits, targets).item()
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    
    return correct_pred.__float__() / num_examples * 100, cross_entropy / num_examples

start_time = time.time()
train_acc_lst, valid_acc_lst = [], []
train_loss_lst, valid_loss_lst = [], []

for epoch in range(NUM_EPOCHS):
    model.train()

    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()

        cost.backward()

        optimizer.step()
          
        if not batch_idx % 300:
            print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
                f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                f' Cost: {cost:.4f}')
              
    model.eval()    
    with torch.set_grad_enabled(False):
        train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device = DEVICE)
        valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device = DEVICE)
        train_acc_lst.append(train_acc)
        valid_acc_lst.append(valid_acc)

        train_loss_lst.append(train_loss)
        valid_loss_lst.append(valid_loss)

        print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} Train Acc.: {train_acc:.2f}%'
              f' | Validation Acc.: {valid_acc:.2f}%')
              
    elapsed = (time.time() - start_time) / 60
    print(f'Time elapsed: {elapsed:.2f} min')

elapsed = (time.time() - start_time) / 60
print(f'Total Training Time: {elapsed:.2f} min')

torch.save(model.state_dict(), 'VGG19_Model Weights.pth')

fig = plt.figure()
plt.plot(range(1, NUM_EPOCHS + 1), train_loss_lst, label = 'Training Loss')
plt.plot(range(1, NUM_EPOCHS+ 1), valid_loss_lst, label = 'Validation Loss')
plt.title('Model Loss')
plt.legend(loc = 'upper right')
plt.ylabel('Loss')
plt.xlabel('Epoch')
fig.savefig('VGG19 ' + 'Loss.png')


fig = plt.figure()
plt.plot(range(1, NUM_EPOCHS + 1), train_acc_lst, label = 'Training Acc')
plt.plot(range(1, NUM_EPOCHS+ 1), valid_acc_lst, label = 'Validation Acc')
plt.legend(loc = 'lower right')
plt.title('Model Acc')
plt.ylabel('Acc')
plt.xlabel('Epoch')
fig.savefig('VGG19 ' + 'Acc.png')

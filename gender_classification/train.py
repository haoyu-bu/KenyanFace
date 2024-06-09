from dataset import load_data, get_optimizer
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='celeba', help='dataset')
parser.add_argument('--output', type=str, default=None, help='output path')
parser.add_argument('--root', type=str, default='./', help='root directory')
args = parser.parse_args()

# Define dataset parameters
dataset_name = args.dataset
root_dir = args.root
model_dir = args.output # Directory where the model will be saved
os.makedirs(model_dir, exist_ok=True)

# Hyperparameters
num_epochs = 15
batch_size = 32
learning_rate = 0.001
optimizer_name = 'adam'  # or 'sgd'

#------------------------------------------------------------------------------
train_loader, validation_loader = load_data(dataset_name, root_dir, batch_size)

# Load a pre-trained ResNet-34 model
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features

# Change the final fully connected layer to output 2 classes (male and female)
model.fc = nn.Linear(num_ftrs, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(model, optimizer_name, learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print("Start training")
best_val_acc = 0.0  # to track the best validation accuracy
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        #print(outputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    scheduler.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    print(f'Epoch {epoch}/{num_epochs - 1} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)

    val_loss = val_running_loss / len(validation_loader.dataset)
    val_acc = val_running_corrects.double() / len(validation_loader.dataset)
    print(f'Epoch {epoch}/{num_epochs - 1} Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    # Check for best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = model.state_dict()
        torch.save(best_model_wts, os.path.join(model_dir, 'best_model.pth'))
        print('Best model saved with accuracy: {:.4f}'.format(best_val_acc))

# Save the model
torch.save(model.state_dict(), os.path.join(model_dir, 'last_model.pth'))

print('Training complete')

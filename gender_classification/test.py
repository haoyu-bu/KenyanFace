import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn as nn
import os
from dataset import FairfaceDataset
import numpy as np

def load_model(model_path):
    model = models.resnet34(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model



def test_model(model, test_loader, criterion, device):
    running_corrects = 0
    running_loss = 0.0

    all_preds = []
    all_labels = []
    all_races = []
    with torch.no_grad():
        for inputs, labels, races in test_loader:
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds += preds.cpu().tolist()
            all_labels += labels.cpu().tolist()
            all_races += races

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects.double() / len(test_loader.dataset)

    from sklearn.metrics import confusion_matrix
    distinct_races = set(all_races)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_races = np.array(all_races)
    for r in distinct_races:
        print(r, sum(all_races == r))
        y_true = all_labels[all_races == r]
        y_pred = all_preds[all_races == r]
        matrix = confusion_matrix(y_true, y_pred)
        print(matrix.diagonal()/matrix.sum(axis=1))

    return total_loss, total_acc

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='celeba', help='path of the model to be tested.')
    args = parser.parse_args()

    # Define paths
    model_path = args.model
    test_csv = 'fairface/fairface_labels.csv'
    test_dir = 'fairface/'

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path)
    model = model.to(device)

    # Define test data transforms and loader
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = FairfaceDataset(csv_file=test_csv, root_dir=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Test the model
    test_loss, test_acc = test_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

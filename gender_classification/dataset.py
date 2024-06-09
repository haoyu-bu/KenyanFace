import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch import Generator
import torch.optim as optim
import torch

gender2num = {'Male': 0, 'Female': 1}

class FairfaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.face_frame = pd.read_csv(csv_file)[['file', 'gender', 'race']]
        self.face_frame['gender'] = self.face_frame['gender'].apply(lambda x : gender2num[x])
        print(len(self.face_frame))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.face_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.face_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.face_frame.iloc[idx, 1]    
        race = self.face_frame.iloc[idx, 2]    

        if self.transform:
            image = self.transform(image)

        return image, label, race

class UTKFaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.face_frame = pd.read_csv(csv_file)[['ID', 'gender']]
        print(len(self.face_frame))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.face_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.face_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.face_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

class UTKFaceAugDataset(Dataset):
    def __init__(self, csv_file, csv_file_aug, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            csv_file_aug (string): Path to the aug csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.face_frame = pd.read_csv(csv_file)
        self.face_frame_aug = pd.read_csv(csv_file_aug)
        self.face_frame = pd.concat([self.face_frame[['ID', 'gender']], self.face_frame_aug[['ID', 'gender']]])
        print("# Samples:", len(self.face_frame))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.face_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.face_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.face_frame.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label

class UTKFaceCelebaDataset(Dataset):
    def __init__(self, csv_file, csv_file_aug, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            csv_file_aug (string): Path to the aug csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.face_frame = pd.read_csv(csv_file)
        self.face_frame_aug = pd.read_csv(csv_file_aug)
        self.face_frame = pd.concat([self.face_frame[['ID', 'gender']], self.face_frame_aug[['ID', 'gender']]])
        print("# Samples:", len(self.face_frame))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.face_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.face_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.face_frame.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label

class CelebaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.face_frame = pd.read_csv(csv_file)[['ID', 'gender']]
        print(len(self.face_frame))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.face_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.face_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.face_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

class CelebaAugDataset(Dataset):
    def __init__(self, csv_file, csv_file_aug, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            csv_file_aug (string): Path to the aug csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.face_frame = pd.read_csv(csv_file)
        self.face_frame_aug = pd.read_csv(csv_file_aug)
        self.face_frame = pd.concat([self.face_frame[['ID', 'gender']], self.face_frame_aug[['ID', 'gender']]])
        print("# Samples:", len(self.face_frame))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.face_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.face_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.face_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(dataset_name, root_dir, batch_size=32, validation_split=0.1, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if dataset_name == 'utkface':
        dataset = UTKFaceDataset(csv_file='UTKface_inthewild/utk.csv', root_dir=root_dir, transform=transform)
    elif dataset_name == 'utkface-aug':
        dataset = UTKFaceAugDataset(csv_file='UTKface_inthewild/utk.csv', csv_file_aug='kenya/sample_20k.csv', root_dir=root_dir, transform=transform)
    elif dataset_name == 'utkface-celeba':
        dataset = UTKFaceCelebaDataset(csv_file='UTKface_inthewild/utk.csv', csv_file_aug='celeba/celeba_20k.csv', root_dir=root_dir, transform=transform)
    elif dataset_name == 'celeba':
        dataset = CelebaDataset(csv_file='celeba/celeba.csv', root_dir=root_dir, transform=transform)
    elif dataset_name == 'celeba-aug':
        dataset = CelebaAugDataset(csv_file='celeba/celeba_100k.csv', csv_file_aug='kenya/sample_100k.csv', root_dir=root_dir, transform=transform)
    else:
        raise ValueError(f'Dataset {dataset_name} not recognized.')

    # Splitting the dataset into train and validation sets
    train_size = int((1 - validation_split) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], Generator().manual_seed(42))

    # Creating data loaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, validation_loader

def get_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f'Optimizer {optimizer_name} not recognized.')
   
    return optimizer

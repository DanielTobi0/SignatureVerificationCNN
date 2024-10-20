import pandas as pd
import numpy as np
import ast

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

train_data_path = 'data/train_data.csv'
test_data_path = 'data/test_data.csv'

data_transform = transforms.Compose([
    # Resize the images to 64x64
    # transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    # transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor()  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])


class CustomDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.df = pd.read_csv(data_path)
        self.df['images'] = self.df['images'].apply(lambda x: np.array(ast.literal_eval(x)))  # Convert back to array
        self.transform = transform

    def __getitem__(self, index: int):
        label = self.df.iloc[index]['labels']
        img = self.df.iloc[index]['images']

        img = img.reshape(224, 224)  #
        img = np.stack([img] * 3)  # 3 channels
        # img = np.expand_dims(img, axis=0) # 1 channel

        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.df)


def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: None,
        batch_size: int
):
    train_data = CustomDataset(train_data_path)
    test_data = CustomDataset(test_data_path)

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )
    return train_dataloader, test_dataloader

# train_dataloader, test_dataloader = create_dataloaders(train_data_path, test_data_path, data_transform, 32)

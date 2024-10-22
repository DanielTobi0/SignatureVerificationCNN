import pandas as pd
import numpy as np
import ast
import torch
from torch.utils.data import Dataset, DataLoader
from utils import data_transform


class CustomDataset(Dataset):
    """
    A custom Dataset class to mimic pytorch datasets functionality. This class takes in a csv file,
    applies transformation to the features and returns the length and data points when indexed
    """

    def __init__(self, data_path: str, transform=None):
        """
        Reads csv file and transform the 'images' feature, from list to numpy array.

        Args:
            data_path (str): Path to the csv file, containing images in arrays.
        """
        self.df = pd.read_csv(data_path)
        self.df['images'] = self.df['images'].apply(lambda x: np.array(ast.literal_eval(x)))  # Convert back to array
        self.transform = transform

    def __getitem__(self, index: int):
        """
        Gets the image and label of a specific index.

        Args:
            index (int): An integer pointing to a data row
        Return:
            Tuple: (image, label)
        """
        label = self.df.iloc[index]['labels']
        img = self.df.iloc[index]['images']

        img = img.reshape(224, 224)
        img = np.stack([img] * 3)  # 3 channels (C, H, W)
        # img = np.expand_dims(img, axis=0) # 1 channel

        img = torch.tensor(img, dtype=torch.float32)
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.df)


def create_dataloaders(
        train_csv_data_path: str,
        test_csv_data_path: str,
        batch_size: int,
):
    """
    A simple function, create dataloaders for train and test datasets.

    Args:
        train_csv_data_path (str): Path to the train data in csv file.
        test_csv_data_path (str): Path to the test data in csv file.
        batch_size (int): Number of sample(s) in a dataloader.

    Returns:
        Tuple: (train_dataloader, test_dataloader)

    Examples:
        # >>> train_dataloader, test_dataloader = create_dataloaders(
                                                    train_csv_data_path,
                                                    test_csv_data_path,
                                                    32)

    """
    train_data = CustomDataset(train_csv_data_path, transform=data_transform)
    test_data = CustomDataset(test_csv_data_path, transform=data_transform)

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
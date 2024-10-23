import numpy as np 
import cv2
import os 
import glob

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class SiameseSignatureDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the signature images.
            transform (callable, optional): Optional transform to be applied on an image.
            mode (string): Whether the dataset is in 'train' or 'test' mode.
        """
        self.root_dir = os.path.join(root_dir, mode)
        self.person_folders = [folder for folder in os.listdir(self.root_dir) if not folder.endswith('_forg')]
        self.transform = transform

    def __len__(self):
        return len(self.person_folders)

    def __getitem__(self, idx):
        """
        Returns a tuple (image1, image2, label) where:
        - image1, image2: Images of the two signatures.
        - label: 1 if similar (both real or both forged), 0 if dissimilar.
        """
        # Select a person's real and forged signature folders
        person_folder = self.person_folders[idx]
        real_folder = os.path.join(self.root_dir, person_folder)
        forg_folder = os.path.join(self.root_dir, person_folder + '_forg')
        
        real_signatures = os.listdir(real_folder)
        forged_signatures = os.listdir(forg_folder)

        # Decide randomly whether to create a positive (similar) or negative (dissimilar) pair
        is_positive_pair = random.choice([True, False])

        if is_positive_pair:
            # Positive pair: two real signatures or two forged signatures from the same person
            if random.choice([True, False]):
                # Pick two real signatures from the same person
                img1_name, img2_name = random.sample(real_signatures, 2)
                img1_path = os.path.join(real_folder, img1_name)
                img2_path = os.path.join(real_folder, img2_name)
            else:
                # Pick two forged signatures from the same person
                img1_name, img2_name = random.sample(forged_signatures, 2)
                img1_path = os.path.join(forg_folder, img1_name)
                img2_path = os.path.join(forg_folder, img2_name)

            label = 1  # Similar pair

        else:
            # Negative pair: one real and one forged signature from the same person
            img1_name = random.choice(real_signatures)
            img2_name = random.choice(forged_signatures)
            img1_path = os.path.join(real_folder, img1_name)
            img2_path = os.path.join(forg_folder, img2_name)

            label = 0  # Dissimilar pair

        # Load the images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


'''def dataloader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    root_dir = "raw_data"

    # Create dataset instances
    train_dataset = SiameseSignatureDataset(root_dir=root_dir, transform=transform, mode='train')
    test_dataset = SiameseSignatureDataset(root_dir=root_dir, transform=transform, mode='test')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader'''
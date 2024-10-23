import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data_setup import SiameseSignatureDataset
from model_builder import SiameseSignatureCNN, SiameseNetwork, ContrastiveLoss


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = SiameseSignatureDataset(root_dir='raw_data', transform=transform, mode='train')
test_dataset = SiameseSignatureDataset(root_dir='raw_data', transform=transform, mode='test')

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

signature_cnn = SiameseSignatureCNN(input_channels=3)
siamese_network = SiameseNetwork(signature_cnn)
contrastive_loss = ContrastiveLoss()
optimizer = torch.optim.Adam(siamese_network.parameters(), lr=0.001)


num_epochs = 15
for epoch in range(num_epochs):
    siamese_network.train()
    for img1, img2, label in train_loader:
        optimizer.zero_grad()

        distance = siamese_network(img1, img2)
        loss = contrastive_loss(distance, label)
        
        loss.backward()
        optimizer.step()
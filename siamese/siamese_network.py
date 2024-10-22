import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data_setup import SiameseSignatureDataset
from model_builder import SiameseSignatureCNN, SiameseNetwork, ContrastiveLoss





# Define transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create dataset instances
train_dataset = SiameseSignatureDataset(root_dir='raw_data', transform=transform, mode='train')
test_dataset = SiameseSignatureDataset(root_dir='raw_data', transform=transform, mode='test')

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Instantiate the base model
signature_cnn = SiameseSignatureCNN(input_channels=3)

# Instantiate the Siamese Network with the base model
siamese_network = SiameseNetwork(signature_cnn)

# Instantiate the contrastive loss
contrastive_loss = ContrastiveLoss()

# Example training loop
optimizer = torch.optim.Adam(siamese_network.parameters(), lr=0.001)


num_epochs = 15

for epoch in range(num_epochs):
    siamese_network.train()
    for img1, img2, label in train_loader:
        optimizer.zero_grad()

        # Forward pass through the Siamese network
        distance = siamese_network(img1, img2)

        # Compute the contrastive loss
        loss = contrastive_loss(distance, label)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
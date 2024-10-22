import torch
from typing import List, Tuple
from siamese_network import siamese_network, contrastive_loss
from data_setup import dataloader


train_dataloader, test_dataloader = dataloader()

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    num_epochs = 15
    threshold = 0.4  # You can tune this based on your dataset

    for epoch in range(num_epochs):
        siamese_network.train()  # Set the network to training mode
        total_loss = 0.0  # To accumulate total loss for this epoch
        correct_predictions = 0  # To count correct predictions for accuracy
        total_pairs = 0  # To keep track of the number of pairs

        for img1, img2, label in dataloader:
            optimizer.zero_grad()

            # Forward pass through the Siamese network
            distance = siamese_network(img1, img2)

            # Compute the contrastive loss
            loss = contrastive_loss(distance, label)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss for the epoch
            total_loss += loss.item()

            # Calculate accuracy
            predicted_labels = (distance < threshold).float()  # Predict label based on threshold
            correct_predictions += (predicted_labels == label).sum().item()  # Compare with actual labels
            total_pairs += label.size(0)  # Total number of pairs processed

        # Compute average loss and accuracy for this epoch
        avg_loss = total_loss / len(dataloader)  # Average loss
        train_accuracy = correct_predictions / total_pairs  # Accuracy for this epoch

        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')
        return avg_loss, train_accuracy
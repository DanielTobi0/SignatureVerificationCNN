"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import argparse
import os
import torch
import data_setup, engine, model_builder, utils
from torchvision import transforms


# Set up argument parser
parser = argparse.ArgumentParser(description="Hyperparameter settings for training the model.")
parser.add_argument('--num_epochs', type=int, default=15, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--hidden_units', type=int, default=10, help='Number of hidden units in the model')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')

# Parse the arguments
args = parser.parse_args()


# Setup hyperparameters from args
# NUM_EPOCHS = 15
# BATCH_SIZE = 32
# HIDDEN_UNITS = 10
# LEARNING_RATE = 0.001

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate

# Setup directories
train_dir = "raw_data/train"
test_dir = "raw_data/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.SignatureCNN(
    input_channels=3
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="signature_model.pth")



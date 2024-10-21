import argparse
import torch
import data_setup, engine, model_builder, utils
from file_paths import file_path
from utils import device

# Set up argument parser
parser = argparse.ArgumentParser(description="Hyperparameter settings for training the model.")
parser.add_argument('--num_epochs', type=int, default=15, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--hidden_units', type=int, default=10, help='Number of hidden units in the model')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
args = parser.parse_args()


NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate


# Dataloaders
train_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_csv_data_path=file_path['train_csv_data_path'],
    test_csv_data_path=file_path['test_csv_data_path'],
    batch_size=BATCH_SIZE
)

# Model
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
                 model_name=file_path["signature_model.pth"])



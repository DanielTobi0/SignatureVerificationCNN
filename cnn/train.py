import argparse
import torch
import data_setup, engine, model_builder, utils
from file_paths import file_path
from torch.optim.lr_scheduler import ReduceLROnPlateau



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
).to(file_path['device'])

# Set loss and optimizer
class_weights = torch.tensor([1.0, 3.0])  # Higher weight for minority class (e.g., forged signatures)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=1e-4)

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             scheduler=scheduler,
             epochs=NUM_EPOCHS,
             device=file_path['device'])

utils.save_model(model=model,
                 target_dir="models",
                 model_name=file_path["model_name"])


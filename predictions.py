import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import List, Tuple
from utils import load_model

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Predict on a target image with a target model
def pred_and_plot_image(
        model: torch.nn.Module,
        image_path: str,
        image_size: Tuple[int, int] = (224, 224),
        transform: torchvision.transforms = None,
        device: torch.device = device,

):
    """Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        image_path (str): Filepath to target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
    """

    # Open image
    img = os.listdir(image_path)[0]
    img_path = f'{image_path}/{img}'
    img = Image.open(img_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

    ### Predict on image ###

    # Make sure the model is on the target device
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    if target_image_pred_label == 0:
        print('Signature matched')
    elif target_image_pred_label == 1:
        print('Signature do not match')
    else:
        print('Something is wrong')

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)


cnn_model = load_model(model_path='models/signature_model.pth')
pred_and_plot_image(image_path='predict', model=cnn_model)

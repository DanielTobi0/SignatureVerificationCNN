import torch
import torchvision
from torchvision import transforms
import os
from typing import Tuple
from utils import load_model
from file_paths import file_path
from grad_cam import predict_outcome
import cv2


def pred_and_plot_image(
        model: torch.nn.Module,
        image_path: str,
        image_size: Tuple[int, int] = (90, 90),
        transform: torchvision.transforms = None,
        device: torch.device = file_path['device'],

):
    """
    Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        image_path (str): Filepath to target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
    
    Returns:
        None: Prints the result of the prediction ("Signature matched" or "Signature do not match") and calls a 
              visualization function predict_outcome() .

    Example Usage:
        pred_and_plot_image(model=trained_model, image_path="path/to/image", device=device)
    """

    img = os.listdir(image_path)[0]
    img_path = f'{image_path}/{img}'

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        )

    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    if target_image_pred_label == 0:
        print('Signature matched')
    elif target_image_pred_label == 1:
        print('Signature do not match')
    else:
        print('Something is wrong')

    predict_outcome()


cnn_model = load_model(model_path=file_path['model_save_path'])
pred_and_plot_image(image_path=file_path['predict_image_path'], model=cnn_model, device=file_path['device'])
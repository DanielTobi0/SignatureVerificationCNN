import os
import glob
import cv2
import numpy as np
import pandas as pd
from file_paths import file_path
'''
# download and unzip data to folder
!unzip /content/drive/MyDrive/SignatureRecognition/ML_material.zip -d /content/drive/MyDrive/SignatureRecognition

# remove duplicated data-folder and unused folder
duplicated_path = '/content/drive/MyDrive/SignatureRecognition/ML_material/sign_data/sign_data'
if os.path.exists(duplicated_path):
  !rm -rf  /content/drive/MyDrive/SignatureRecognition/ML_material/sign_data/sign_data
  !rm -rf /content/drive/MyDrive/SignatureRecognition/__MACOSX
'''


def prepare_data(data_dir: str):
    """
    Extract image data from a specified folder, applies transformations, and save to a Pandas Dataframe.
    data_dir:
        person1_real_signatures
        person1_forged_signatures
        ...
        personX_real_signatures
        personX_forged_signatures

    Args:
        data_dir (str): Folder path containing images organized in subdirectories.

    Raises:
        FileNotFoundError: if the provided 'data_dir' does not exist.
        ValueError: if no images are found in the data folder

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - images (np.ndarray): A flattened array of processed image data
            - labels (np.ndarray): An array of labels corresponding to the images (0: real, 1: forged).
    Examples:
        # >>> images, labels = prepare_data('path/to/data')

    """

    SIZE = 224

    real_images = []
    forged_images = []

    for per in os.listdir(data_dir):
        for data in glob.glob(os.path.join(data_dir, per, '*.*')):
            img = cv2.imread(data)

            # skip if the image cannot be read
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (SIZE, SIZE))  # numpy requires that all elements to have shape in each dimension.

            if per[-1] == 'g':
                forged_images.append(img)
            else:
                real_images.append(img)

    if not real_images and not forged_images:
        raise ValueError('No images found in the specified directory.')

    real_images = np.array(real_images)
    forged_images = np.array(forged_images)

    real_labels = np.zeros((real_images.shape[0], 1))
    forged_labels = np.ones((forged_images.shape[0], 1))

    # Concatenate the real and forged images and labels
    images = np.concatenate((real_images, forged_images))
    labels = np.concatenate((real_labels, forged_labels))

    # Flatten the images
    images = images.reshape(images.shape[0], -1)

    # Save data to pandas dataframe
    df = pd.DataFrame({
        'images': [img.tolist() for img in images],
        'labels': labels.flatten()})

    clean_data = 'data/'
    if not os.path.exists(clean_data):
        os.makedirs(clean_data)

    # Save dataframe # changed from 66:
    df.to_csv(f'{clean_data}{data_dir[8:]}_data.csv', index=False)

    return images, labels


train_images, train_labels = prepare_data(file_path['raw_train_dir'])
test_images, test_labels = prepare_data(file_path['raw_test_dir'])

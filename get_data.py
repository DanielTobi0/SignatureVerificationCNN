# Import required libraries
import os
import glob
import random
import ast
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
# unzip data to folder
!unzip /content/drive/MyDrive/SignatureRecognition/ML_material.zip -d /content/drive/MyDrive/SignatureRecognition

# remove duplicated data-folder and un-used folder
duplicated_path = '/content/drive/MyDrive/SignatureRecognition/ML_material/sign_data/sign_data'
if os.path.exists(duplicated_path):
  !rm -rf  /content/drive/MyDrive/SignatureRecognition/ML_material/sign_data/sign_data
  !rm -rf /content/drive/MyDrive/SignatureRecognition/__MACOSX
'''


def prepare_data(data_dir: str):
    """Args:
    data_dir:

  """
    SIZE = 224

    real_images = []
    forged_images = []

    for per in os.listdir(data_dir):
        for data in glob.glob(os.path.join(data_dir, per, '*.*')):
            img = cv2.imread(data)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (SIZE, SIZE))  # numpy requires that all elements to have shape in each dimension.

            if per[-1] == 'g':
                forged_images.append(img)
            else:
                real_images.append(img)

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


# be careful of this
# train_dir = "/content/drive/MyDrive/SignatureRecognition/ML_material/sign_data/train"
# test_dir = "/content/drive/MyDrive/SignatureRecognition/ML_material/sign_data/test"

train_dir = "raw_data/train"
test_dir = "raw_data/test"
train_images, train_labels = prepare_data(train_dir)
test_images, test_labels = prepare_data(test_dir)

'''
train_images.shape

# Function to display random images
def show_random_images(images, labels):
    plt.figure(figsize=(10, 4))

    # Get a random sample of 9 indices
    random_indices = random.sample(range(len(images)), k=3)

    for i, idx in enumerate(random_indices):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[idx].reshape(224, 224), cmap='gray')
        plt.title(f"Label: {int(labels[idx])}")
        plt.axis("off")

    plt.show()


# Display images from the train dataset
df = pd.read_csv('/content/drive/MyDrive/SignatureRecognition/data/test_data.csv')
df['images'] = df['images'].apply(lambda x: np.array(ast.literal_eval(x)))  # Convert back to array

show_random_images(df['images'], df['labels'])'''

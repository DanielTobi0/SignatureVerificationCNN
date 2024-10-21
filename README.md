# Signature Recognition Project

## How to Run the Code
Follow these steps to set up and run the code for the Signature Recognition task.

### 1. Set Up a Virtual Environment
Create a virtual environment and activate it:
```
# create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### 2. Install required packages
Install the necessary dependencies using the requirements.txt file
```
pip install -r requirements.txt
```

### 3. Prepare the Data Directory
In the parent directory of your virtual environment, create a folder named ```raw_data```
```
mkdir raw_data
```

### 4. Download and Organize the Dataset
Download the zip file containing the images from this [link](https://drive.google.com/drive/folders/1696vpZtrG5QudnkM4o_L66kslxBIVQin?usp=drive_link)

Download Images
- Unzip the downloaded file.
- Navigate to the directory: ```ML_Material >> sign_data```.
- Copy the ```train and test``` folders to the ```raw_data``` folder you created earlier.

### 5. Convert Images to Numpy Arrays
Run the ```get_data.py``` file. This step only needs to be done once. It converts the images to a numpy array and stores them in a CSV file
```
python get_data.py
```

### 6. Train the Model
To train the model, you can pass hyperparameter arguments via the terminal. Here’s an example command
```
python train.py --num_epochs 15 --batch_size 32 --hidden_units 10 --learning_rate 0.001
```

### 7. Make Predictions
To predict the authenticity of an image:

Create a folder named ```predict``` in the parent directory.
Place the image you want to predict in the ```predict``` folder.
Run the ```prediction.py``` file
```
python prediction.py
```

### 8. Grad Cam
To visualize the grad cam, the part the model focuses when making prediction.
Run the ```grad_cam.py```

import torch

file_path = {
    'raw_train_dir': "raw_data/train",
    'raw_test_dir': "raw_data/test",
    'train_csv_data_path': 'data/train_data.csv',
    'test_csv_data_path': 'data/test_data.csv',
    'model_save_path': 'models/signature_model.pth',
    'predict_image_path': 'predict',
    'model_name': "signature_model.pth",
    'predict_image_folder': 'predict',
    'image_size': 224,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

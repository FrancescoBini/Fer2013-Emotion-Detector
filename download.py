import os
from datasets import load_dataset
import gdown
import zipfile
import kagglehub

# WORKING VERSION - DOWNLOAD
# Download latest version
# path = kagglehub.dataset_download("msambare/fer2013")

# train_path = "/Users/francesco/.cache/kagglehub/datasets/msambare/fer2013/versions/1/train"
# test_path = "/Users/francesco/.cache/kagglehub/datasets/msambare/fer2013/versions/1/test"


## GOOGLE DRIVE
# https://drive.google.com/file/d/10TolZhzTxuy7IDHHcg6-C02_2MfrnmKq/view?usp=sharing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "train")
test_path  = os.path.join(BASE_DIR, "test")

if not os.path.exists(train_path) or not os.path.exists(test_path):
    url = "https://drive.google.com/uc?id=10TolZhzTxuy7IDHHcg6-C02_2MfrnmKq"
    zip_path = os.path.join(BASE_DIR, "fer2013.zip")
    
    print("Downloading FER2013 dataset...")
    gdown.download(url, zip_path, quiet=False)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)
    
    print("Done! Dataset is ready.")
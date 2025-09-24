import kagglehub
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
from io import BytesIO
from datasets import load_dataset

# WORKING VERSION - DOWNLOAD
# Download latest version
# path = kagglehub.dataset_download("msambare/fer2013")

# train_path = "/Users/francesco/.cache/kagglehub/datasets/msambare/fer2013/versions/1/train"
# test_path = "/Users/francesco/.cache/kagglehub/datasets/msambare/fer2013/versions/1/test"


# Get the base directory of your project (where download.py lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to the train and test folders relative to your project
train_path = os.path.join(BASE_DIR, "train")
test_path = os.path.join(BASE_DIR, "test")

# Optional: check if directories exist
if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Train or test folder not found. Make sure they exist in the project root.")

print("Train directory:", train_path)
print("Test directory:", test_path)
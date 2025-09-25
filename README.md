**Facial Expression Recognition (FER2013) – CNN & ResNet**

This project implements a Facial Expression Recognition (FER2013) pipeline in PyTorch.
It allows you to train different CNN architectures on the dataset, experiment with transformations, and customize training parameters.

**Repository Structure**

**download.py**
Handles dataset setup. The FER2013 dataset is downloaded from Google Drive.
Users only need to run this script once to set up the dataset locally.

**models.py**
Contains model definitions:
SimpleCNN → A lightweight CNN built with nn.Conv2d, fast and efficient.
SimpleResNet → A ResNet-inspired architecture with residual blocks (skip connections).
ResNet18 → A fully implemented ResNet18

**train.py**
Core training script.
- Loads the dataset and applies transformations.
- Lets the user choose for extra augmentations (random rotations & flips) for dataset enrichment.
- Lets the user choose which model to run (SimpleCNN, SimpleResNet or ResNet18).
- Lets the user choose for epoch number (integer)
- Lets the user choose for learning rate (float)
- Trains the model with chosen hyperparameters.
- Prints training and validation results after each epoch.
- Optionally displays sample predictions (true vs. predicted emotions).

Example Run
python download.py       # Download dataset
python models.py         # States models
python train.py          # Train the chosen model


During training, you’ll see logs like:

Epoch [5/30], Loss: 0.9616, Acc: 0.6441
Validation: Loss: 1.2783, Acc: 0.5297

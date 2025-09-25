from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import SimpleCNN, SimpleCNN2, SimpleResNet, ResNet18
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import random
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(BASE_DIR, "train")
test_path  = os.path.join(BASE_DIR, "test")


# Transformations

# Convert to grayscale 
transform = transforms.Compose([
    transforms.Grayscale(),           # ensure 1-channel
    transforms.Resize((48, 48)),      # just in case
    transforms.ToTensor(),            # converts [0,255] -> [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # scales to [-1,1]
])

# Load datasets
train_dataset = ImageFolder(root=train_path, transform=transform)
test_dataset  = ImageFolder(root=test_path, transform=transform)

classes = test_dataset.classes

# MORE TRANSFORMATIONS - flips, rotate etc
train_transform = transforms.Compose([
    transforms.Grayscale(),                        # ensure 1-channel
    transforms.Resize((48, 48)),                  # resize to 48x48
    transforms.RandomHorizontalFlip(),            # random horizontal flip
    transforms.RandomRotation(10),                # random rotation ±10 degrees
    transforms.RandomResizedCrop(48, scale=(0.9,1.0)),  # slight zoom / crop
    transforms.ToTensor(),                         # convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])   # scale to [-1,1]
])

# Validation / test transformations (no augmentation)
val_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

trans_train = ImageFolder(train_path, transform=train_transform)
trans_test = ImageFolder(test_path, transform=val_transform)


# DataLoaders
batch_size = 32

transf = input('Do you want to use further transformations like rotation, flip etc? (yes/no): ')
if transf == 'no':
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('Just converting to Grayscale')
else:
    train_loader = DataLoader(trans_train, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(trans_test, batch_size=batch_size, shuffle=False)
    print('Using Flip, Rotation etc')


# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# for images, labels in train_loader:
#     print("Batch shape:", images.shape)
#     break

# Device
if torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# print("Using device:", device)

# List of valid inputs
valid_inputs = ["Simple", "ResNet", "Res18"]

while True:
    model_name = input('Choose model, either ResNet, Res18, or Simple (CNN using nn.Conv2d): ')
    if model_name in valid_inputs:
        print(f"Valid input received: {model_name}")
        break
    else:
        print("Invalid input. Try again.")


# model_name = input('Choose model between ResNet, simple2 (CNN with nn.Conv2d), or simple (CNN with built Conv layer): ' )
num_epochs = int(input('Input amount of epochs: '))
learning_rate = float(input('Input learning rate: '))


## CHOOSE THE MODEL
# model_name = "ResNet" 

if model_name == "Simple":
    model = SimpleCNN2(num_classes=7)
elif model_name == "ResNet":
    model = SimpleResNet(num_classes=7)
elif model_name == "Res18":
    model = ResNet18(num_classes=7)

model = model.to(device)


# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

# Training loop
# num_epochs = 3
print_every = 400

print(model)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % print_every == 0:  # every 10 batches
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")


    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    model.eval()  # switch to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # no gradients needed for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_acc = correct / total

    print(f"Validation: Loss: {avg_test_loss:.4f}, Acc: {test_acc:.4f}")


model.eval()

## showcasing model performance with images
# Take 10 random indices from validation dataset
indices = random.sample(range(len(test_dataset)), 10)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i, idx in enumerate(indices):
    img, label = test_dataset[idx] # get image and true label
    img_batch = img.unsqueeze(0).to(device)  # add batch dimension and move to device

    with torch.no_grad():
        output = model(img_batch)
        pred_label = torch.argmax(output, dim=1).item()

    # Convert tensor to numpy for plotting
    img_np = img.squeeze().cpu().numpy()  # remove channel dim if grayscale
    img_np = (img_np * 0.5) + 0.5  # unnormalize if using Normalize(mean=0.5, std=0.5)

    axes[i].imshow(img_np, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f"True: {classes[label]}\nPred: {classes[pred_label]}")

plt.tight_layout()
plt.show()
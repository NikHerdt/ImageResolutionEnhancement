import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from torchvision import transforms
import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# Set up data split
image_paths = pd.read_csv('/Users/herdt/manifest-1729788671964/image_paths.csv')

# Output directory
output_dir = '/Users/herdt/manifest-1729788671964/output_training_files'

# Split data into training and testing
train_paths, test_paths = train_test_split(image_paths, test_size=0.2)

# Set up data loader
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths.iloc[idx]['Save Path']  # Use the 'input_image' column
        image = pydicom.dcmread(image_path).pixel_array
        image = apply_voi_lut(image, pydicom.dcmread(image_path))
        image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = ImageDataset(train_paths, transform)
test_dataset = ImageDataset(test_paths, transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set up model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers except the last one
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 1)

# Set up loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
model.train()

for epoch in range(5):
    for images in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, torch.ones(images.size(0), 1))  # Assuming target is a tensor of ones
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))

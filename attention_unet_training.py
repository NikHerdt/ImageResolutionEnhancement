import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from skimage.transform import resize

output_save_path = "/Users/herdt/manifest-1729788671964/output_training_files"

# Define the data
df = pd.read_csv("/Users/herdt/manifest-1729788671964/image-paths.csv")
train_data_path = df['training-images']
test_data_path = df['testing-images']

# Turn the data into a tensor
def load_image(path):
    try:
        img = Image.open(path).convert('RGB')  # Ensure 3 channels
    except FileNotFoundError:
        print(f"Image not found: {path}")
        return None
    img_array = np.array(img)
    max_val = np.max(img_array)
    if max_val == 0:
        max_val = 1  # Prevent division by zero
    normalized_img_array = img_array / max_val

    new_size = (256, 256)
    resized_img = resize(normalized_img_array, new_size, anti_aliasing=True)

    img = torch.tensor(resized_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Change to (C, H, W)

    return img

# Load the data
train_data = [img for path in train_data_path if (img := load_image(path)) is not None]
test_data = [img for path in test_data_path if (img := load_image(path)) is not None]

# Define the U-Net model 
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        
        # Contracting path (Encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Expanding path (Decoder)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoding
        x1 = self.encoder(x)
        x2 = self.bottleneck(x1)
        x3 = self.decoder(x2)
        # Resize output to match input dimensions
        x3_resized = F.interpolate(x3, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x3_resized


unet_model = UNet()

# Train the U-Net model
unet_criterion = nn.MSELoss()
unet_optimizer = optim.Adam(unet_model.parameters(), lr=0.001)

for epoch in range(50):
    for data in train_data:
        unet_optimizer.zero_grad()
        output = unet_model(data)
        loss = unet_criterion(output, data)
        loss.backward()
        unet_optimizer.step()
        print(f"Epoch {epoch}, Loss {loss.item()}")

# Save the U-Net model
torch.save(unet_model.state_dict(), f"{output_save_path}/unet_model.pth")

# Test the U-Net model
unet_model.eval()

unet_loss = 0
for data in test_data:
    output = unet_model(data)
    unet_loss += unet_criterion(output, data).item()

unet_loss /= len(test_data)
print(f"U-Net Loss: {unet_loss}")

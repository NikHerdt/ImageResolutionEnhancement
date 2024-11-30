import torch
import numpy as np
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import pandas as pd

output_save_path = "/Users/herdt/manifest-1729788671964/output_training_files"

# Define the data
df = pd.read_csv("/Users/herdt/manifest-1729788671964/image-paths.csv")
train_data_path = df['training-images']
test_data_path = df['testing-images']

# Turn the data into a tensor
def load_image(path):
    img = Image.open(path)
    img = img.resize((256, 256))
    img = np.array(img)
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img, dtype=torch.float32)
    return img

# Load the data
train_data = [load_image(path) for path in train_data_path]
test_data = [load_image(path) for path in test_data_path]

# Define the resnet model 
class DenoisingResNet(nn.Module):
    def __init__(self):
        super(DenoisingResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(256, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256 * 256 * 3)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 256, 256, 3)
        return x

resnet_model = DenoisingResNet()

# Define the gan model
class DenoisingGAN(nn.Module):
    def __init__(self):
        super(DenoisingGAN, self).__init__()
        self.generator = DenoisingResNet()
        self.discriminator = models.resnet18(pretrained=False)
        self.discriminator.conv1 = nn.Conv2d(256, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.discriminator.fc = nn.Linear(self.discriminator.fc.in_features, 1)

    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)
        return x
    
gan_model = DenoisingGAN()

# Train the resnet model
resnet_criterion = nn.MSELoss()
resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)

for epoch in range(100):
    for data in train_data:
        resnet_optimizer.zero_grad()
        output = resnet_model(data)
        loss = resnet_criterion(output, data)
        loss.backward()
        resnet_optimizer.step()
        print(f"Epoch {epoch}, Loss {loss.item()}")

# Train the gan model
gan_criterion = nn.BCEWithLogitsLoss()
gan_optimizer = optim.Adam(gan_model.parameters(), lr=0.001)

for epoch in range(100):
    for data in train_data:
        gan_optimizer.zero_grad()
        output = gan_model(data)
        loss = gan_criterion(output, torch.ones_like(output))
        loss.backward()
        gan_optimizer.step()
        print(f"Epoch {epoch}, Loss {loss.item()}")

# Save the models
torch.save(resnet_model.state_dict(), f"{output_save_path}/resnet_model.pth")
torch.save(gan_model.state_dict(), f"{output_save_path}/gan_model.pth")

# Test the models
resnet_model.eval()
gan_model.eval()

resnet_loss = 0
gan_loss = 0

for data in test_data:
    output = resnet_model(data)
    resnet_loss += resnet_criterion(output, data).item()
    output = gan_model(data)
    gan_loss += gan_criterion(output, torch.ones_like(output)).item()

resnet_loss /= len(test_data)
gan_loss /= len(test_data)

print(f"ResNet Loss: {resnet_loss}")
print(f"GAN Loss: {gan_loss}")
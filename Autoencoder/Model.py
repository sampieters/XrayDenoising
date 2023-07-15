from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import Data
from torch.utils.data import DataLoader, random_split
from torchsummary import summary

#  configuring device
if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('Running on the GPU')
else:
  device = torch.device('cpu')
  print('Running on the CPU')

input_size = (256, 1248, 1)

class ConvolutionalAutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            #torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

data = Data.Data()

train_set = data.read_from_folder('../DFFCinput/noisy', '../DFFCinput/perfect')

train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_set, valid_set = random_split(train_set, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=16)
val_loader = DataLoader(valid_set, batch_size=16)

test_set = data.read_from_folder('../DFFCinput/noisy', '../DFFCinput/perfect')
test_loader = DataLoader(train_set, batch_size=16)

model = ConvolutionalAutoEncoder()

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

epochs = 50
train_losses = []
val_losses = []
min_valid_loss = np.inf

for epoch in range(epochs):
    train_loss = 0.0
    for data, labels in tqdm(train_loader):
        # Transfer Data to GPU if available
        data, labels = data.to(device), labels.to(device)

        # Clear the gradients
        optimizer.zero_grad()
        # Forward Pass
        target = model(data)
        # Find the Loss
        loss = loss_function(target, labels)
        # Calculate gradients
        loss.backward()
        # Update Weights
        optimizer.step()
        # Calculate Loss
        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()  # Optional when not using Model Specific layer
    for data, labels in train_loader:
        # Transfer Data to GPU if available
        data, labels = data.to(device), labels.to(device)

        # Forward Pass
        target = model(data)
        # Find the Loss
        loss = loss_function(target, labels)
        # Calculate Loss
        valid_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(valid_loss / len(val_loader))
    print(f'Epoch [{epoch + 1}/{epochs}] \n'
          f'\tTraining Loss: { train_loss / len(train_loader)} \n'
          f'\tValidation Loss: { valid_loss / len(val_loader)}')

    if min_valid_loss > valid_loss:
        print(f'\tValidation loss decreased({min_valid_loss:.6f} -> {valid_loss:.6f}) so model is saved')
        min_valid_loss = valid_loss

        # Saving State Dict
        torch.save(model.state_dict(), '../output/info/checkpoint.pth')

# Test the model

model.load_state_dict(torch.load('../output/info/checkpoint.pth'))
model.eval()

type = '{0:04d}'
j = 0
with torch.no_grad():
    for data in tqdm(test_loader):
        images, labels = data

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        image_tensors = torch.split(outputs, 1, dim=0)

        # iterate over the image tensors and print their shapes
        for i, image_tensor in enumerate(image_tensors):
            array = image_tensor.numpy()
            array = array.squeeze()
            array = array.reshape((256, 1248))
            array[array < 0] = 0
            array = np.round((2 ** 16 - 1) * array).astype(np.uint16)
            Data.imwrite(array, f"../output/autoencoder/denoised_{type.format(j)}.tif")
            j += 1

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(range(epochs), train_losses)
plt.plot(range(epochs), val_losses)
plt.savefig('../output/info/Loss.png')
plt.close()
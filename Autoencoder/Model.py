from matplotlib import pyplot as plt
from torchsummary import summary
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import Data

input_size = (256, 1248)
training_dir = '../input/training'
perfect_dir = '../input/perfect'
training_perc = 0.8
validation_perc = 0.1
test_perc = 0.1
batch_size = 16
lr = 0.001
epochs = 40
weight_decay = 0
type = '{0:04d}'
output_dir = '../output/autoencoder'
output_info = '../output/info'
details = False

#  configuring device, transfer data to GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    device_type = 'GPU'
else:
    device = torch.device('cpu')
    device_type = 'CPU'


class ConvolutionalAutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


data = Data.Data(batch_size)
train_set = data.read_from_folder(training_dir, perfect_dir)
train_loader, val_loader, test_loader = data.rand_split(train_set, training_perc, validation_perc, test_perc)

model = ConvolutionalAutoEncoder()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

if details:
    summary(model, input_size, batch_size, device)

train_losses = []
val_losses = [np.inf]
for epoch in range(1, epochs + 1):
    train_loss = 0.0
    print(f'Epoch [{epoch}/{epochs}]')
    for noisy, perfect in tqdm(train_loader):
        noisy, perfect = noisy.to(device), perfect.to(device)
        # Clear the gradients
        optimizer.zero_grad()
        # Forward pass
        target = model(noisy)
        # Find the loss
        loss = loss_function(target, perfect)
        # Calculate gradients
        loss.backward()
        # Update weights
        optimizer.step()
        # Calculate loss
        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()
    for noisy, perfect in train_loader:
        noisy, perfect = noisy.to(device), perfect.to(device)
        # Forward pass
        target = model(noisy)
        # Find and calculate the loss
        loss = loss_function(target, perfect)
        valid_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(valid_loss / len(val_loader))
    print(f'\tTraining Loss: {train_losses[-1]} \n'
          f'\tValidation Loss: {val_losses[-1]}')

    if min(val_losses[:-1]) > val_losses[-1]:
        print(f'\tValidation loss decreased({min(val_losses[:-1]):.6f} -> {val_losses[-1]:.6f}) so model is saved')
        torch.save(model.state_dict(), f'{output_info}/Checkpoint.pth')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(range(1, epoch + 1), train_losses)
    plt.plot(range(1, epoch + 1), val_losses[1:])
    plt.savefig(f'{output_info}/Loss.png')
    plt.close()

# Test the model

# model.load_state_dict(torch.load('../output/info/checkpoint.pth'))
# model.eval()

j = 0
with torch.no_grad():
    for noisy, perfect in tqdm(test_loader):
        noisy, perfect = noisy.to(device), perfect.to(device)
        outputs = model(noisy)
        _, predicted = torch.max(outputs.data, 1)
        image_tensors = torch.split(outputs, 1, dim=0)

        for image_tensor in image_tensors:
            array = image_tensor.numpy()
            array = array.squeeze()
            array = array.reshape(input_size)
            array[array < 0] = 0
            array = np.round((2 ** 16 - 1) * array).astype(np.uint16)
            Data.imwrite(array, f"{output_dir}/denoised_{type.format(j)}.tif")
            j += 1

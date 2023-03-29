import numpy
import torch
import torch.nn as nn
import Data
from tqdm import tqdm

#  configuring device
if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('Running on the GPU')
else:
  device = torch.device('cpu')
  print('Running on the CPU')


class ConvolutionalAutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Define the input tensor size
        input_size = (256, 1248, 1)

        # Define the number of output channels
        out_channels = 3

        # Define the kernel size
        kernel_size = (3, 3)

        # Define the stride
        stride = (1, 1)

        # Define the padding
        padding = (1, 1)

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_size[2], out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=out_channels, out_channels=input_size[2], kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

data = Data.Data()
data.read_from_folder('../input/noisy')

model = ConvolutionalAutoEncoder()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e8)

epochs = 10
outputs = []
losses = []



for epoch in range(epochs):
    for (image, _) in tqdm(data.trainloader):

      #image = image.reshape(-1, 28*28)

        reconstructed = model(image)

        loss = loss_function(reconstructed, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss)
        #outputs.append((epochs, image, reconstructed))
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
print(numpy.average(loss))
#plt.style.use('fivethirtyeight')
#plt.xlabel('Iterations')
#plt.ylabel('Loss')
#plt.plot(losses[-100:])
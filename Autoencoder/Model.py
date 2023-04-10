import numpy
import numpy as np
import torch
import torch.nn as nn
import Data
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image as im

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
data.read_from_folder('../input/training', '../input/perfect')

model = ConvolutionalAutoEncoder()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e8)

epochs = 3
outputs = []
losses = []

id_noisy = 0
id_perfect = 0

for epoch in range(epochs):
    dataloader_iterator = iter(data.train_perfect_loader)
    for (noisy_image, label) in tqdm(data.train_noisy_loader):
        perfect_image = next(dataloader_iterator)

        reconstructed = model(noisy_image)

        loss = loss_function(reconstructed, noisy_image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        outputs.append((epochs, noisy_image, reconstructed))
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {numpy.average(losses).item():.4f}")
#plt.style.use('fivethirtyeight')
#plt.xlabel('Iterations')
#plt.ylabel('Loss')
#plt.plot(losses[-100:])

print('Finished training')

# Test the model
correct = 0
total = 0
type = '{0:04d}'

j = 0
with torch.no_grad():
    for data in data.testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()
        image_tensors = torch.split(outputs, 1, dim=0)

        # iterate over the image tensors and print their shapes
        for i, image_tensor in enumerate(image_tensors):
            array = image_tensor.numpy()
            array = array.squeeze()
            array = array.reshape((256, 1248))

            array[array < 0] = 0
            #tmp = -np.log(tmp)
            #tmp[np.isinf(tmp)] = 10 ** 5
            #array = np.round((2 ** 16 - 1) * array).astype(np.uint16)

            im.fromarray(array).save(f"../output/autoencoder/denoised_{type.format(j)}.tif")
            j += 1

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
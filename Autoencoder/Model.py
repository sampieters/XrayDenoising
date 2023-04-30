import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image as im
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import numpy
import torch
import Data

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
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            #torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.ReLU(),
            #torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.ReLU(),
            #torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.ReLU(),
            #torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            #torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.ReLU(),
            #torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.ReLU(),
            #torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.ReLU(),
            #torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

data = Data.Data()
train_noisy_set = data.read_from_folder('../input/training')
train_perfect_set = data.read_from_folder('../input/perfect')
test_set = data.read_from_folder('../input/denoise_testing')
train_noisy_loader, train_perfect_loader, val_noisy_loader, val_perfect_loader = data.random_split(train_noisy_set, train_perfect_set)
test_loader, _, _, _ = data.random_split(test_set, test_set, 1)

model = ConvolutionalAutoEncoder()
loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

epochs = 3
outputs = []
losses = []
avg_losses = []
for epoch in range(epochs):
    dataloader_iterator = iter(train_perfect_loader)
    for (noisy_image, label) in tqdm(train_noisy_loader):
        perfect_image = next(dataloader_iterator)
        reconstructed = model(noisy_image)

        loss = loss_function(reconstructed, perfect_image[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        outputs.append((epochs, noisy_image, reconstructed))
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {numpy.average(losses).item():.4f}")
    avg_losses.append(numpy.average(losses).item())

    if epoch == 1:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, '../output/info/checkpoint.pth')


#model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
#with torch.no_grad():
#    dataloader_iterator = iter(val_perfect_loader)
#    for (noisy_image, label) in tqdm(val_noisy_loader):
#        perfect_image = next(dataloader_iterator)
#        reconstructed = model(noisy_image)

#        loss = loss_function(reconstructed, perfect_image[0])

#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()

#        losses.append(loss.item())
#        outputs.append((epochs, noisy_image, reconstructed))

print('Finished training \nStarted testing')

# Test the model
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
            im.fromarray(array).save(f"../output/autoencoder/denoised_{type.format(j)}.tif")
            j += 1

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(range(epochs), avg_losses)
plt.savefig('../output/info/Loss.png')
plt.close()
import torch
import torch.nn as nn

#  configuring device
if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('Running on the GPU')
else:
  device = torch.device('cpu')
  print('Running on the CPU')



class Encoder(nn.model):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.ReLU(self.conv1(x))
        return x


class Decoder(nn.model):
    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channels=3, out_channels=16, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.linear(x)
        return x


class ConvolutionalAutoEncoder(nn.model):
    def __init__(self, encoder, decoder):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.encoder.to(device)

        self.decoder = decoder
        self.decoder.to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
from Autoencoder.Model import ConvolutionalAutoEncoder
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchsummary import summary
from tqdm import tqdm
import numpy as np
import torch
import Data

input_size = (256, 1248)                        # the dimensions of the input projections
training_dir = '../input/simulated_one/training/'   # directory where the noisy training data is saved
outPrefixFFC = 'AUTOENCODER'                    # prefix of the autoencoder corrected projections
perfect_dir = '../input/simulated_one/perfect/'     # directory where the perfect training data is saved
training_perc = 0.8                             # training percentage, amount of data in dataset that is used for training
validation_perc = 0.1                           # validation percentage, amount of data in dataset that is used for validation
test_perc = 0.1                                 # test percentage, amount of data in dataset that is used for testing
batch_size = 32                                 # number of samples that are propagated through the autoencoder at once
lr = 0.001                                      # beginning learning rate
epochs = 100                                     # number of times the entire dataset is passed through the model during training
weight_decay = 0
numType = '04d'                                 # number type used in image names
fileFormat = '.tif'                             # image format
output_dir = '../output/autoencoder/'           # Directory where the autoencoder corrected projections are saved
details = False                                 # turn on(True)/off(False) the details of the model


def write_info(content):
    file_path = "../output/autoencoder/info/info.txt"
    with open(file_path, "w") as file:
        file.write(content)
    file.close()

# configuring device, transfer data to GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    device_type = 'GPU'
else:
    device = torch.device('cpu')
    device_type = 'CPU'

# Get all images and divide them into training, validation and test set
data = Data.Data(batch_size)
train_set = data.read_from_folder(training_dir, perfect_dir, False)
train_loader, val_loader, test_loader = data.rand_split(train_set, training_perc, validation_perc, test_perc)

# Choose the model, loss fucntion and optimizer
model = ConvolutionalAutoEncoder()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

if details:
    summary(model, input_size, batch_size, device)

content = f"Device type: {device_type}\n" \
              f"Model: \n" \
              f"Number of epochs: {epochs}\n" \
              f"Starting learning rate: {lr}\n" \
              f"Weight decay: {weight_decay}\n" \
              f"Batch size: {batch_size}\n" \
              f"Train set percentage: {training_perc}\n" \
              f"Validation set percentage: {validation_perc}\n" \
              f"Test set percentage: {test_perc}\n"
write_info(content)


# Train the model
def train(checkpoint=None):
    start_epoch = 0
    # Load model and optimizer states and epoch from the checkpoint
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']

    train_losses = [np.inf]
    val_losses = [np.inf]
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        print(f'Epoch [{start_epoch + epoch}/{start_epoch + epochs}]')
        model.train()
        for noisy, perfect in tqdm(train_loader):
            # Transfer data to GPU if possible
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
        for noisy, perfect in val_loader:
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
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'{output_dir}info/Checkpoint.pth')

        plt.plot(range(1, epoch + 1), train_losses[1:])
        plt.plot(range(1, epoch + 1), val_losses[1:])
        plt.show()

    plt.savefig(f'{output_dir}info/Loss.png')
    plt.close()

    train_content = f"Training losses:\n"
    val_content = f"\nValidation losses:\n"
    epoch = 0
    for index in range(len(train_losses)):
        train_content += f"Epoch {epoch + 1}: {train_losses[index]}\n"
        val_content += f"Epoch {epoch + 1}: {val_losses[index]}\n"
        epoch += 1
    write_info(train_content + val_content)


# Test the model
def tst(test_dir=None):
    global test_loader
    checkpoint = torch.load('../output/autoencoder/info/Checkpoint.pth')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    if test_dir is not None:
        test_set = data.read_from_folder(test_dir)
        test_loader = DataLoader(test_set, batch_size=batch_size)

    with torch.no_grad():
        j = 0
        for noisy, perfect in tqdm(test_loader):
            noisy, perfect = noisy.to(device), perfect.to(device)
            outputs = model(noisy)
            _, predicted = torch.max(outputs.data, 1)
            image_tensors = torch.split(outputs, 1, dim=0)

            for image_tensor in image_tensors:
                array = image_tensor.numpy()
                array = array.squeeze()
                array = array.reshape(input_size)
                array = np.round((2 ** 16 - 1) * array).astype(np.uint16)
                Data.imwrite(array, output_dir + outPrefixFFC + f'{j:{numType}}' + fileFormat)
                j += 1

train()
tst()



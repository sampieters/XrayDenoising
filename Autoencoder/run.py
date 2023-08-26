from Autoencoder.Model import ConvolutionalAutoEncoder
from Autoencoder.Data import Data, imwrite
from Autoencoder.summary import summary
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import os


def write_info(path, content):
    with open(path, "a+") as file:
        file.write(content)
    file.close()


def run(param):
    # configuring device, transfer data to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        device_type = 'cuda'
    else:
        device = torch.device('cpu')
        device_type = 'cpu'

    # Make info directory for training
    directory = os.path.dirname(f'{param["AUTOENCODER"]["outDir"]}info/')
    os.makedirs(directory, exist_ok=True)

    # Get all images and divide them into training, validation and test set
    data = Data(param["AUTOENCODER"]["batchSize"])
    train_set = data.read_from_folder(param["AUTOENCODER"]["trainDir"], param["AUTOENCODER"]["perfDir"], False)
    train_loader, val_loader, test_loader = data.rand_split(train_set, param["AUTOENCODER"]["trainPerc"], param["AUTOENCODER"]["valPerc"], param["AUTOENCODER"]["testPerc"])

    model = ConvolutionalAutoEncoder()
    only_test = False
    if not only_test:
        # Choose the model, loss function and optimizer

        loss_function = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=param["AUTOENCODER"]["lr"], weight_decay=param["AUTOENCODER"]["weightDecay"])

        content = f'Device type: {device_type}\n' \
                  f'Model:\n{summary(model, (1, param["size"][0], param["size"][1]), param["AUTOENCODER"]["batchSize"], device_type)}\n' \
                  f'Number of epochs: {param["AUTOENCODER"]["epochs"]}\n' \
                  f'Starting learning rate: {param["AUTOENCODER"]["lr"]}\n' \
                  f'Weight decay: {param["AUTOENCODER"]["weightDecay"]}\n' \
                  f'Batch size: {param["AUTOENCODER"]["batchSize"]}\n' \
                  f'Train set percentage: {param["AUTOENCODER"]["trainPerc"]}\n' \
                  f'Validation set percentage: {param["AUTOENCODER"]["valPerc"]}\n' \
                  f'Test set percentage: {param["AUTOENCODER"]["testPerc"]}\n'
        write_info(f'{param["AUTOENCODER"]["outDir"]}info/info.txt', content)

        # Train the model
        start_epoch = 0
        # Load model and optimizer states and epoch from the checkpoint
        if param["AUTOENCODER"]["checkpoint"] is not None:
            checkpoint = torch.load(param["AUTOENCODER"]["checkpoint"])
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch']

        train_losses = [np.inf]
        val_losses = [np.inf]
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        for epoch in range(1, param["AUTOENCODER"]["epochs"] + 1):
            train_loss = 0.0
            print(f'Epoch [{start_epoch + epoch}/{start_epoch + param["AUTOENCODER"]["epochs"]}]')
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
                torch.save(checkpoint, f'{param["AUTOENCODER"]["outDir"]}info/Checkpoint.pth')

            plt.plot(range(1, epoch + 1), train_losses[1:], label='Training Loss')
            plt.plot(range(1, epoch + 1), val_losses[1:], label='Validation Loss')
            plt.legend()
            plt.savefig(f'{param["AUTOENCODER"]["outDir"]}info/Loss.png')
            plt.show()
        plt.close()

        train_content = f"----------------------------------------------------------------\n" \
                        f"Training losses:\n"
        val_content = f"----------------------------------------------------------------\n" \
                      f"Validation losses:\n"
        epoch = 0
        for index in range(len(train_losses)):
            train_content += f"Epoch {epoch + 1}: {train_losses[index]}\n"
            val_content += f"Epoch {epoch + 1}: {val_losses[index]}\n"
            epoch += 1
        write_info(f'{param["AUTOENCODER"]["outDir"]}info/info.txt', train_content + val_content)

    # Test the model
    checkpoint = torch.load(f'{param["AUTOENCODER"]["outDir"]}info/Checkpoint.pth')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    if param["AUTOENCODER"]["test"]:
        test_set = data.read_from_folder(param["inDir"])
        test_loader = DataLoader(test_set, batch_size=param["AUTOENCODER"]["batchSize"])

    with torch.no_grad():
        j = 1
        for noisy, perfect in tqdm(test_loader):
            noisy, perfect = noisy.to(device), perfect.to(device)
            outputs = model(noisy)
            _, predicted = torch.max(outputs.data, 1)
            image_tensors = torch.split(outputs, 1, dim=0)

            for image_tensor in image_tensors:
                array = image_tensor.numpy()
                array = array.squeeze()
                array = array.reshape(param["size"])
                array = param["bit"](np.round((np.iinfo(param["bit"]).max * array)))
                imwrite(array, param["AUTOENCODER"]["outDir"] + param["AUTOENCODER"]["outPrefix"] + f'{j:{param["numType"]}}' + param["fileFormat"])
                j += 1

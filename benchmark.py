import torch

from FFC.Python.ConventionalFlatFieldCorrection import ConventionalFlatFieldCorrection
from DFFC.Python.DynamicFlatFieldCorrection import DynamicFlatFieldCorrection
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
import torch.nn.functional as F
from Utils.Utils import imread
import numpy as np


def check(parameters, algorithm):
    """
    This function loads pairs of images, one from a check directory and one from the specified algorithm's output
    directory, and calculates the Mean Squared Error (MSE) loss between them for multiple images. The mean MSE loss is
    then computed and returned.

    Parameters:
    :param parameters: A dictionary containing various parameters and file paths required for image comparison.
    :param algorithm: A string containing the name of the algorithm being checked.
    :return: The mean MSE of the pairs of images
    """
    meanloss = 0
    for i in range(parameters["nrProj"]):
        check1 = imread(f'{parameters["checkDir"]}{parameters["prefixProj"]}{parameters["firstProj"] + i:{parameters["numType"]}}{parameters["fileFormat"]}')
        check1 = check1 / np.iinfo(parameters["bit"]).max
        check2 = imread(f'{parameters[algorithm]["outDir"]}{parameters[algorithm]["outPrefix"]}{parameters["firstProj"] + i:{parameters["numType"]}}{parameters["fileFormat"]}')
        check2 = check2 / np.iinfo(parameters["bit"]).max

        totensor = ToTensor()
        check1 = totensor(check1)
        check2 = totensor(check2)

        loss = F.mse_loss(check1, check2)
        loss = 20 * torch.log10(1.0 / torch.sqrt(loss))
        meanloss += loss
    print(f'Mean MSE Loss: {meanloss / parameters["nrProj"]}')
    return meanloss / parameters["nrProj"]

def benchmark_all(parameters):
    """
    This function performs benchmarking of the FFC, DFFC and CAE algorithms for different iterations of white prior and
    post-processing. It iteratively adjusts the white prior and post-processing parameters, runs both algorithms,
    calculates Mean Squared Error (MSE) losses, and plots the comparison results.

    :param parameters: A dictionary containing various parameters and file paths required for benchmarking.
    """
    y_values_FFC = []
    y_values_DFFC = []

    real_white_prior = parameters["nrWhitePrior"]
    real_white_post = parameters["nrWhitePost"]

    parameters["nrWhitePost"] = 0
    parameters["nrWhitePrior"] = 0
    x_values = list(range(10, (real_white_prior + real_white_post) % 10))
    for i in range(10, real_white_prior + real_white_post, 10):
        if real_white_prior > i:
            parameters["nrWhitePost"] = i
        else:
            parameters["nrWhitePrior"] = i - real_white_prior
        ConventionalFlatFieldCorrection(parameters)
        DynamicFlatFieldCorrection(parameters)
        FFC_loss = check(parameters, "FFC")
        DFFC_loss = check(parameters, "DFFC")
        y_values_FFC.append(FFC_loss)
        y_values_DFFC.append(DFFC_loss)

    # Create a figure and axis
    plt.figure()

    # Plotting the data
    plt.plot(x_values, y_values_FFC, label='FFC')
    plt.plot(x_values, y_values_DFFC, label='DFFC')

    # Adding labels and title
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.title('MSE Loss Comparison of Denoising Algorithms')

    # Adding legend
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()
    plt.savefig("plot.png")


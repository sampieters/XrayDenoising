from FFC.Python.ConventionalFlatFieldCorrection import ConventionalFlatFieldCorrection
from DFFC.Python.DynamicFlatFieldCorrection import DynamicFlatFieldCorrection
from Autoencoder.run import run

import argparse
import os
from PIL import Image as im
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToTensor

parameters = {
    "inDir":             "./input/benchmark/noisy0/",
    "checkDir":          "./input/benchmark/perfect0/",
    "FFC": {                                                # ## FFC ###
        "outDir":        "./output/FFC/Python/",            # directory the FFC corrected projections are saved
        "outPrefix":     "FFC_",                            # prefix of the FFC corrected projections
    },
    "DFFC": {                                               # ## DFFC ###
        "outDir":        "./output/DFFC/Python/",           # directory the DFFC corrected projections are saved
        "outPrefix":     "DFFC_",                           # prefix of the DFFC corrected projections
        "downsample":    2,                                 # amount of downsampling during dynamic flat field estimation (integer between 1 and 20)
        "nrPArepetitions": 10,                              # number of parallel analysis repetitions
    },
    "AUTOENCODER": {                                        # ## AUTOENCODER ###
        "outDir":        "./output/autoencoder/",           # directory the autoencoder corrected projections are saved
        "outPrefix":     "AUTOENCODER_",                    # prefix of the autoencoder corrected projections
        "trainDir":      "./input/simulated/training/", # directory the noisy training data is saved
        "perfDir":       "./input/simulated/perfect/",  # directory the perfect training data is saved
        "checkpoint":    None,                              # path to load a checkpoint to train the model from (None = no checkpoint)
        "test":          True,                              # test on input directory if True else on trainDir
        "trainPerc":     0.8,                               # training percentage, amount of data that is used for training
        "valPerc":       0.1,                               # validation percentage, amount of data that is used for validation
        "testPerc":      0.1,                               # test percentage, amount of data that is used for testing (value between 0 and 1)
        "batchSize":     32,                                # number of samples propagated through the autoencoder at once
        "lr":            0.001,                             # starting learning rate
        "epochs":        100,                               # number of times the entire dataset is passed through the model during training
        "weightDecay":   0,                                 #
    },
    "prefixProj":        "dbeer_5_5_",                      # prefix of the projections
    "prefixFlat":        "dbeer_5_5_",                      # prefix of the flatfields
    "prefixDark":        "dbeer_5_5_",                      # prefix of the darkfields
    "numType":           "04d",                             # number type used in image names
    "fileFormat":        ".tif",                            # image format
    "bit":               np.uint16,                         # bit size of a pixel
    "size":              (256, 1248),                       # the dimensions of the input projections
    "firstDark":         1,                                 # image number of first dark field
    "nrDark":            20,                                # number of dark fields
    "firstWhitePrior":   21,                                # image number of first prior flat field
    "nrWhitePrior":      300,                               # number of white (flat) fields BEFORE acquiring the projections
    "firstWhitePost":    572,                               # image number of first post flat field
    "nrWhitePost":       0,                                 # number of white (flat) fields AFTER acquiring the projections
    "firstProj":         321,                               # image number of first projection
    "nrProj":            50,                                # number of acquired projections
    "scale":             [0, 2]                             # output images are scaled between these values
}


def imread(path):
    image = im.open(path)
    image = np.asarray(image).astype(np.double)
    image = image / np.iinfo(parameters["bit"]).max
    return image

def check(algorithm):
    meanloss = 0
    for i in range(parameters["nrProj"]):
        check1 = imread(f'{parameters["checkDir"]}{parameters["prefixProj"]}{parameters["firstProj"] + i:{parameters["numType"]}}{parameters["fileFormat"]}')
        check2 = imread(f'{parameters[algorithm]["outDir"]}{parameters[algorithm]["outPrefix"]}{parameters["firstProj"] + i:{parameters["numType"]}}{parameters["fileFormat"]}')

        totensor = ToTensor()
        check1 = totensor(check1)
        check2 = totensor(check2)

        mse_loss = F.mse_loss(check1, check2)
        meanloss += mse_loss
        print(f"MSE Loss: {mse_loss}")
    print(f'Mean MSE Loss: {meanloss / parameters["nrProj"]}')

def run_command(algorithms):
    if 'FFC' in algorithms:
        ConventionalFlatFieldCorrection(parameters)
    if 'DFFC' in algorithms:
        DynamicFlatFieldCorrection(parameters)
    if 'AUTOENCODER' in algorithms:
        run(parameters)
    print(f"The algorithms have been executed successfully.")


def benchmark_command(algorithms):
    if 'FFC' in algorithms:
        ConventionalFlatFieldCorrection(parameters)
        check('FFC')
    if 'DFFC' in algorithms:
        DynamicFlatFieldCorrection(parameters)
        check('DFFC')
    if 'AUTOENCODER' in algorithms:
        run(parameters)
        check('AUTOENCODER')
    print(f"The algorithms have been executed and tested successfully.")


def clear_command():
    directory_paths = [parameters["FFC"]["outDir"],
                      parameters["DFFC"]["outDir"],
                      parameters["AUTOENCODER"]["outDir"],
                      parameters["AUTOENCODER"]["trainDir"],
                      parameters["AUTOENCODER"]["perfDir"]
                      ]
    try:
        for directory in directory_paths:
            # Check if the directory exists
            if not os.path.exists(directory):
                print(f"The directory '{directory}' does not exist.")
                return

            # Remove all files and subdirectories within the directory
            for root, _, files in os.walk(directory, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)

            # Remove all empty subdirectories (if any)
            for root, dirs, _ in os.walk(directory, topdown=False):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)

            print(f"The folder '{directory}' has been cleared successfully.")
    except Exception as e:
        print(f"An error occurred while clearing the folder: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, choices=["run", "simulate", "benchmark", "clear"], help="The actions")
    algo_group = parser.add_argument_group("Run command arguments")
    algo_group.add_argument("--algorithms", nargs='+', type=str, help="Algorithms to run")

    args = parser.parse_args()

    if args.action == "run":
        # Handle the "run" command
        if args.algorithms is not None:
            run_command(args.algorithms)
        else:
            print(f"The '{args.action}' command requires '--algorithms' argument.")
    elif args.action == "simulate":
        simulate_command()
    elif args.action == "benchmark":
        # Handle the "benchmark" command
        if args.algorithms is not None:
            benchmark_command(args.algorithms)
        else:
            print(f"The '{args.action}' command requires '--algorithms' argument.")
    elif args.action == "clear":
        # Handle the "clear" command
        clear_command()

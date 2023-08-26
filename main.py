from FFC.Python.ConventionalFlatFieldCorrection import ConventionalFlatFieldCorrection
from DFFC.Python.DynamicFlatFieldCorrection import DynamicFlatFieldCorrection
from Simulation.simulate import make_benchmark_dataset, make_training_dataset
from Autoencoder.run import run
from benchmark import check
import numpy as np
import argparse
import time
import os

parameters = {
    "inDir":             "./input/benchmark/noisy/",
    "checkDir":          "./input/benchmark/perfect/",
    "FFC": {                                                # ## FFC ###
        "outDir":        "./output/FFC/Python/",            # directory the FFC corrected projections are saved
        "outPrefix":     "FFC_",                            # prefix of the FFC corrected projections
    },
    "DFFC": {                                               # ## DFFC ###
        "outDir":        "./output/DFFC/Python/",           # directory the DFFC corrected projections are saved
        "outPrefix":     "DFFC",                           # prefix of the DFFC corrected projections
        "downsample":    2,                                 # amount of downsampling during dynamic flat field estimation (integer between 1 and 20)
        "nrPArepetitions": 10,                              # number of parallel analysis repetitions
    },
    "AUTOENCODER": {                                        # ## AUTOENCODER ###
        "outDir":        "./output/autoencoder/",           # directory the autoencoder corrected projections are saved
        "outPrefix":     "AUTOENCODER_",                    # prefix of the autoencoder corrected projections
        "trainDir":      "./input/simulated/noisy/",        # directory the noisy training data is saved
        "perfDir":       "./input/simulated/perfect/",      # directory the perfect training data is saved
        "checkpoint":    "./output/autoencoder/info/Checkpoint.pth",                              # path to load a checkpoint to train the model from (None = no checkpoint)
        "test":          True,                              # test on input directory if True else on trainDir
        "trainPerc":     0.8,                               # training percentage, amount of data that is used for training
        "valPerc":       0.1,                               # validation percentage, amount of data that is used for validation
        "testPerc":      0.1,                               # test percentage, amount of data that is used for testing (value between 0 and 1)
        "batchSize":     32,                                # number of samples propagated through the autoencoder at once
        "lr":            0.001,                            # starting learning rate
        "epochs":        30,                                # number of times the entire dataset is passed through the model during training
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
    "nrWhitePost":       300,                               # number of white (flat) fields AFTER acquiring the projections
    "firstProj":         321,                               # image number of first projection
    "nrProj":            250,                               # number of acquired projections
    "scale":             [0, 1]                             # output images are scaled between these values
}


def run_command(algorithms):
    """
    This function takes a list of algorithm names as input and executes the corresponding algorithms based on the provided choices.

    :param algorithms: A list of strings containing the algorithm names to execute, e.g., ['FFC', 'DFFC', 'AUTOENCODER'].

    It can execute the following algorithms:
    - 'FFC': Conventional Flat Field Correction
    - 'DFFC': Dynamic Flat Field Correction
    - 'AUTOENCODER': Convolutional autoencoder-based image processing
    """
    if 'FFC' in algorithms:
        ConventionalFlatFieldCorrection(parameters)
    if 'DFFC' in algorithms:
        DynamicFlatFieldCorrection(parameters)
    if 'AUTOENCODER' in algorithms:
        run(parameters)
    print(f"The algorithms have been executed successfully.")


def simulate_command(option):
    """
    Simulate a dataset based on the provided option, generating data with or without dark- and flatfields.

    :param option: The simulation option, a string which can be 'training' or 'real'.

    This function simulates a dataset based on the provided option:
    - 'training': Generates a training dataset without dark- and flatfields.
    - 'real': Generates a real dataset containing dark- and flatfields.
    """
    if option == "training":
        make_training_dataset(parameters)
    elif option == "real":
        make_benchmark_dataset(parameters)
    print(f"Simulation '{option}' finished successfully.")


def benchmark_command(algorithms, version):
    """
    Execute and benchmark specified image processing algorithms based on user-defined choices.

    Parameters:
    :param algorithms: A list of strings with algorithm names to execute and benchmark, e.g., ['FFC', 'DFFC', 'AUTOENCODER'].
    :param version: The version as string of algorithms to use ('Python', 'MATLAB', '').

    This function takes a list of algorithm names and a version as input and executes the corresponding algorithms based
    on the provided choices.
    It can execute and benchmark the following algorithms:
    - 'FFC': Conventional Flat Field Correction
    - 'DFFC': Dynamic Flat Field Correction
    - 'AUTOENCODER': convolutional autoencoder-based image processing

    The 'version' parameter allows specifying a specific version of algorithms. If 'version' is not provided or set to
    'Python', the Python implementation is used, else MATLAB version.
    """
    if 'FFC' in algorithms:
        if version is None or version == "Python":
            start_time = time.time()
            ConventionalFlatFieldCorrection(parameters)
            end_time = time.time()
            print(f"FFC time: {end_time - start_time}")
        check(parameters, 'FFC')
    if 'DFFC' in algorithms:
        if version is None or version == "Python":
            start_time = time.time()
            DynamicFlatFieldCorrection(parameters)
            end_time = time.time()
            print(f"DFFC time: {end_time - start_time}")
        check(parameters, 'DFFC')
    if 'AUTOENCODER' in algorithms:
        start_time = time.time()
        run(parameters)
        end_time = time.time()
        print(f"convolutional autoencoder time: {end_time - start_time}")
        check(parameters, 'AUTOENCODER')
    if len(algorithms) == 0:
        benchmark_all(parameters)
    print(f"The algorithms have been executed and tested successfully.")


def clear_command():
    """
    Clear specified directories by removing all files and subdirectories.

    This function clears directories specified in the 'directory_paths' list by removing all files and subdirectories
    within them. It is typically used to clean up output directories associated with image processing algorithms.
    """
    # Specify which paths to clear
    directory_paths = [parameters["FFC"]["outDir"],
                      parameters["DFFC"]["outDir"],
                      parameters["AUTOENCODER"]["outDir"],
                      parameters["AUTOENCODER"]["trainDir"],
                      parameters["AUTOENCODER"]["perfDir"]
                      ]
    try:
        # Loop over all directories (if they exist)
        for directory in directory_paths:
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, choices=["run", "simulate", "benchmark", "clear"], help="The actions")
    algo_group = parser.add_argument_group("Run/Benchmark command arguments")
    algo_group.add_argument("--algorithms", nargs='*', type=str, help="Algorithms to run")

    version = parser.add_argument_group("optional benchmark command arguments")
    version.add_argument("--version", type=str, help="version of FFC/DFFC")

    sim_group = parser.add_argument_group("Simulate command arguments")
    sim_group.add_argument("--option", type=str, choices=["training", "real"],
                           help="training = No black/flat-fields, real = Contains black/flat-fields and projections")
    args = parser.parse_args()

    # Check the value of the "--action" argument to determine the action to take
    if args.action == "run":
        # Handle the "run" command
        if args.algorithms is not None:
            run_command(args.algorithms)
        else:
            print(f"The '{args.action}' command requires '--algorithms' argument.")
    elif args.action == "simulate":
        # Handle the "simulate" command
        if args.option is not None:
            simulate_command(args.option)
        else:
            print(f"The '{args.action}' command requires '--algorithms' argument.")
    elif args.action == "benchmark":
        # Handle the "benchmark" command
        if args.algorithms is not None:
            benchmark_command(args.algorithms, args.version)
        else:
            print(f"The '{args.action}' command requires '--algorithms' argument.")
    elif args.action == "clear":
        # Handle the "clear" command
        clear_command()

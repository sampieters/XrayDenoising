# Research Project: X-ray Denoising
This repository contains the code and data for my research project. This project is an extension on an existing project
(link existing project) made in collaboration with Jan Sijbers. 

## Paper
The paper associated with this project can be found [here]({paper_link}).

## Setup
Before running the project, set up an environment (e.g. venv) and install the required libraries listed in the 
'requirements.txt' file. To run a dataset of images, change the values in the parameter directory of the main.py file. 
Every parameter has a comment explaining what it will do.

## Instructions
Because this project includes multiple algorithms to denoise images, multiple commands are included. To run the project
the following command must be used (in the terminal):

`python3 main.py command`

The command field can be replaced with four different commands:

### Run Command:
**Usage:** 
`python3 main.py --action run --algorithms algo1 algo2 ...`

**Description:**
Use this command to run specific algorithms for FFC, DFFC, and Autoencoder to denoise X-ray images.

**Arguments:**
--algorithms: Specify one or more algorithms to run. Available options: "algo1", "algo2", etc.
Example: python main.py --action run --algorithms FFC DFFC Autoencoder

### Simulate Command:
**Usage:** 
`python3 main.py --action simulate --option training/real`

**Description:**
Use this command to simulate scenarios for FFC, DFFC, and Autoencoder with or without black/flat-fields and projections.

**Arguments:**
--option: Specify the simulation option. Available options: "training" (no black/flat-fields) or "real" (contains black/flat-fields and projections).
Example: python main.py --action simulate --option real

### Benchmark Command:
**Usage:**
`python3 main.py --action benchmark --algorithms algo1 algo2 ...`

**Description:**
Use this command to benchmark specific algorithms for FFC, DFFC, and Autoencoder on X-ray image denoising.

**Arguments:**
--algorithms: Specify one or more algorithms to benchmark. Available options: "algo1", "algo2", etc.
Example: python main.py --action benchmark --algorithms FFC DFFC Autoencoder

### Clear Command:
**Usage:**
`python3 main.py --action clear`

**Description:**
Use this command to perform the "clear" action (specific details of the action are not provided in the script).

**Arguments:**
This command has no additional arguments.
Example: python main.py --action clear
Note: Replace "algo1", "algo2", etc. with the actual names of the algorithms used in your project, and update the command descriptions to match the specific functionality of each algorithm. Additionally, provide more details on the "clear" action if it has specific functionalities in your project.

## Structure






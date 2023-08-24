import random
from Simulation.Machine import *

# image size
image_size = (256, 1248)

prefixProj =         'dbeer_5_5_'   # prefix of the original projections
prefixFlat =         'dbeer_5_5_'   # prefix of the flat fields
prefixDark =         'dbeer_5_5_'   # prefix of the dark fields
numType =            '04d'         # number type used in image names
fileFormat =         '.tif'         # image format

type = np.uint16
nrDark =             20             # number of dark fields
firstDark =          1              # image number of first dark field
nrWhitePrior =       300            # number of white (flat) fields BEFORE acquiring the projections
firstWhitePrior =    21             # image number of first prior flat field
nrProj =             251        	# number of acquired projections
firstProj =          321            # image number of first projection

firstWhitePost = 572  # image number of first post flat field
nrWhitePost = 300  # number of white (flat) fields AFTER acquiring the projections

def make_benchmark_dataset(param):
    writeDIR = './input/benchmark/'

    machine = XXX()
    darkfields = machine.generate_darkfields(param["nrDark"])
    flatfields = machine.generate_flatfields(param["nrWhitePrior"])
    clean = machine.generate_projections(nrProj)

    # Make the simulated noisy projections
    projections = np.zeros((nrProj, image_size[0], image_size[1]))
    for i in range(nrProj):
        d_j = darkfields[np.random.randint(0, nrDark)]
        f_j = flatfields[np.random.randint(0, nrWhitePrior)]
        n_j = clean[i]
        projections[i] = np.exp(-n_j) * (f_j - d_j) + d_j

    # Write everything to a file
    for i in range(nrProj):
        imwrite(clean[i], type, writeDIR + 'perfect/' + prefixDark + f'{firstProj + i:{numType}}' + fileFormat)

    for i in range(nrDark):
        imwrite(darkfields[i], type, writeDIR + 'noisy/' + prefixDark + f'{firstDark + i:{numType}}' + fileFormat)

    for i in range(nrWhitePrior):
        imwrite(flatfields[i], type, writeDIR + 'noisy/' + prefixFlat + f'{firstWhitePrior + i:{numType}}' + fileFormat)

    for i in range(nrProj):
        imwrite(projections[i], type, writeDIR + 'noisy/' + prefixProj + f'{firstProj + i:{numType}}' + fileFormat)

    for i in range(nrWhitePost):
        imwrite(flatfields[nrWhitePrior + i], type, writeDIR + 'noisy/' + prefixFlat + f'{firstWhitePost + i:{numType}}' + fileFormat)

def make_training_dataset(param):
    writeDIR = './input/simulated/'

    amount = 600

    machine = XXX()
    darkfields = machine.generate_darkfields(amount)
    flatfields = machine.generate_flatfields(amount)
    clean = machine.generate_projections(amount)

    # Make the simulated noisy projections
    projections = np.zeros((amount, image_size[0], image_size[1]))
    objects = np.zeros((amount, image_size[0], image_size[1]))

    random_dark = random.choices(range(len(darkfields)), k=amount)
    random_flat = random.sample(range(len(flatfields)), amount)
    random_proj = random.sample(range(len(clean)), amount)

    for i in range(amount):
        d_j = darkfields[random_dark[i]]
        f_j = flatfields[random_flat[i]]
        n_j = clean[random_proj[i]]
        objects[i] = n_j
        projections[i] = np.exp(-n_j) * (f_j - d_j) + d_j

    # Write everything to a file
    for i in range(amount):
        imwrite(objects[i], type, writeDIR + 'perfect/' + prefixProj + f'{firstProj + i:{numType}}' + fileFormat)

    for i in range(amount):
        imwrite(projections[i], type, writeDIR + 'noisy/' + prefixProj + f'{firstProj + i:{numType}}' + fileFormat)

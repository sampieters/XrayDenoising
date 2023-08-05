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
nrProj =             50        	    # number of acquired projections
firstProj =          321            # image number of first projection


def make_benchmark_dataset(param):
    writeDIR = './input/benchmark/'

    machine = XXX()
    darkfields = machine.generate_darkfields()
    flatfields = machine.generate_flatfields()
    clean = machine.generate_projections()

    # Make the simulated noisy projections
    projections = np.zeros((nrProj + 1, image_size[0], image_size[1]))
    for i in range(nrProj + 1):
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

def make_training_dataset(param):
    writeDIR = './input/simulated/'

    machine = XXX()
    darkfields = machine.generate_darkfields()
    flatfields = machine.generate_flatfields()
    clean = machine.generate_projections()

    # TODO: parameter to generate how many training data, should maybe not contain duplicates for flatfields in random function
    amount = 300
    # Make the simulated noisy projections
    projections = np.zeros((amount, image_size[0], image_size[1]))
    for i in range(amount):
        d_j = darkfields[np.random.randint(0, nrDark)]
        f_j = flatfields[np.random.randint(0, nrWhitePrior)]
        n_j = clean[i]
        projections[i] = np.exp(-n_j) * (f_j - d_j) + d_j

    # Write everything to a file
    for i in range(amount):
        imwrite(clean[i], type, writeDIR + 'perfect/' + prefixProj + f'{firstProj + i:{numType}}' + fileFormat)

    for i in range(amount):
        imwrite(projections[i], type, writeDIR + 'noisy/' + prefixProj + f'{firstProj + i:{numType}}' + fileFormat)

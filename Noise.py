import noise
import numpy as np

# Set the size of the 2D grid
#width = 1248
#height = 256

# Set the scale of the noise (adjust this to change the frequency of the noise)
#scale = 50.0

# Set the octaves (adjust this to change the complexity of the noise)
#octaves = 6

# Set the persistence (adjust this to change the roughness of the noise)
#persistence = 0.5

# Set the lacunarity (adjust this to change the scale of the noise)
#lacunarity = 2.0

# Set the seed (optional, if you want to generate the same noise pattern each time)
#seed = 10

def generate_perlin_noise_2d(width, height, scale, octaves, persistence, lacunarity, seed):
    if seed is None:
        seed = np.random.randint(0, 100)

    # Generate the Perlin noise using the noise library
    perlin_noise = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            perlin_noise[y][x] = noise.pnoise2(x/scale, y/scale, octaves=octaves, persistence=persistence,
                                               lacunarity=lacunarity, repeatx=width, repeaty=height, base=seed)
    perlin_noise = (perlin_noise - np.min(perlin_noise)) / (np.max(perlin_noise) - np.min(perlin_noise))
    return perlin_noise

def generate_ridged_multifractal_noise(width, height, scale, octaves, persistence, lacunarity, seed):
    # Generate a grid of 2D noise
    perlin_noise = generate_perlin_noise_2d(width, height, scale, octaves, persistence, lacunarity, seed)
    # Apply ridge function
    perlin_noise = np.abs(perlin_noise)

    # Normalize to 0-1 range
    perlin_noise = (perlin_noise - np.min(perlin_noise)) / (np.max(perlin_noise) - np.min(perlin_noise))
    return perlin_noise

def generate_turbulence_noise(width, height, scale, octaves, persistence, lacunarity, seed):
    turbulence_noise = np.zeros((height, width))

    for octave in range(octaves):
        octave_scale = scale * (lacunarity ** octave)
        octave_amplitude = persistence ** octave

        perlin_noise = generate_perlin_noise_2d(width, height, octave_scale, 1, 0.5, 2.0, seed)
        turbulence_noise += octave_amplitude * np.abs(perlin_noise)

    return turbulence_noise

def generate_darkfield(width, height):
    pass


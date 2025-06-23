import numpy as np
import random
from imgaug import augmenters as iaa

def brightness_aug(image, gamma):
    x = np.transpose(image, (2, 0, 1))
    aug_brightness = iaa.GammaContrast(gamma=gamma)
    aug_image = aug_brightness(images=x)
    aug_image = np.transpose(aug_image, (1, 2, 0))

    return aug_image

def poisson_noise(image):
    # noise_level = 20*random.random()# You may need to experiment with this value
    noise_level = 5
    # Generate Poisson noise and add it to the image
    poisson_noise = np.random.poisson(image / 255.0 * noise_level) / noise_level
    noisy_image = (image / 255.0 + poisson_noise).clip(0, 1) * 255.0

    # Convert the noisy image back to uint8 format (if needed)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image):
    """
    Add salt and pepper noise to an image.

    Parameters:
    - image: Input image
    - salt_prob: Probability of adding salt noise
    - pepper_prob: Probability of adding pepper noise

    Returns:
    - Noisy image
    """
    salt_prob = 0.2 * random.random()
    pepper_prob = 0.2 * random.random()
    noisy_image = np.copy(image)

    # Salt noise
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 255

    # Pepper noise
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0

    return noisy_image

def add_gaussian_noise(image):
    """
    Add Gaussian noise to an image.

    Parameters:
    - image: Input image
    - mean: Mean of the Gaussian distribution (default: 0)
    - sigma: Standard deviation of the Gaussian distribution (default: 25)

    Returns:
    - Noisy image
    """
    mean = 0
    sigma = 10 * random.random()
    sigma = 10
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)
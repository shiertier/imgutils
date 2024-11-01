"""
Overview:
    Useful tools to remove adversarial noises, just using opencv library without any models.

    .. image:: adversarial_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the adversarial denoising:

    .. image:: adversarial_benchmark.plot.py.svg
        :align: center

    .. note::
        This tool is inspired from `Huggingface - mf666/mist-fucker <https://huggingface.co/spaces/mf666/mist-fucker>`_.
"""
import random

import cv2
import numpy as np
from PIL import Image

from ..data import load_image


def remove_adversarial_noise(
        image: Image.Image, diameter_min: int = 4, diameter_max: int = 6,
        sigma_color_min: float = 6.0, sigma_color_max: float = 10.0,
        sigma_space_min: float = 6.0, sigma_space_max: float = 10.0,
        b_iters: int = 64,
) -> Image.Image:
    """
    Remove adversarial noise from an image using random bilateral filtering.

    This function applies random bilateral filtering iteratively to reduce adversarial noise
    in the input image.

    :param image: The input image.
    :type image: Image.Image

    :param diameter_min: Minimum diameter for bilateral filtering.
    :type diameter_min: int, optional

    :param diameter_max: Maximum diameter for bilateral filtering.
    :type diameter_max: int, optional

    :param sigma_color_min: Minimum filter sigma in the color space for bilateral filtering.
    :type sigma_color_min: float, optional

    :param sigma_color_max: Maximum filter sigma in the color space for bilateral filtering.
    :type sigma_color_max: float, optional

    :param sigma_space_min: Minimum filter sigma in the coordinate space for bilateral filtering.
    :type sigma_space_min: float, optional

    :param sigma_space_max: Maximum filter sigma in the coordinate space for bilateral filtering.
    :type sigma_space_max: float, optional

    :param b_iters: Number of iterations for bilateral filtering.
    :type b_iters: int, optional

    :return: Image with adversarial noise removed.
    :rtype: Image.Image
    """
    image = load_image(image, mode='RGB', force_background='white')
    img = np.array(image).astype(np.float32)
    y = img.copy()

    # Apply random bilateral filtering iteratively
    for _ in range(b_iters):
        diameter = random.randint(diameter_min, diameter_max)
        sigma_color = random.uniform(sigma_color_min, sigma_color_max)
        sigma_space = random.uniform(sigma_space_min, sigma_space_max)
        y = cv2.bilateralFilter(y, diameter, sigma_color, sigma_space)

    # Clip the values and convert back to uint8 for PIL Image
    output_image = Image.fromarray(y.clip(0, 255).astype(np.uint8))
    return output_image

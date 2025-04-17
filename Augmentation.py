"""
This script applies multiple image augmentation techniques such as flipping,
rotation, skewing, shearing, cropping, and elastic distortion to an image.
The augmented images are saved with corresponding suffixes indicating
the transformation applied.
"""

import argparse
import os
import random
import logging
from typing import Callable, Dict

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_image(image_path: str) -> np.ndarray:
    """ Load an image in RGB format using OpenCV """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, original_path: str, suffix: str) -> str:
    """ Save the image with a suffix in the same directory """
    name, ext = os.path.splitext(os.path.basename(original_path))
    output_filename = f"{name}_{suffix}{ext}"
    output_path = os.path.join(os.path.dirname(original_path), output_filename)
    Image.fromarray(image).save(output_path)
    logger.info("Saved: %s", output_path)
    return output_path


def flip_image(image: np.ndarray) -> np.ndarray:
    """ Flip the image horizontally """
    return cv2.flip(image, 1)


def rotate_image(image: np.ndarray) -> np.ndarray:
    """ Rotate the image by a random angle between -30 and 30 degrees """
    angle = random.uniform(-30, 30)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def skew_image(image: np.ndarray) -> np.ndarray:
    """ Apply a random perspective transformation to skew the image """
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([
        [random.uniform(0, w * 0.2), random.uniform(0, h * 0.2)],
        [w - random.uniform(0, w * 0.2), random.uniform(0, h * 0.2)],
        [random.uniform(0, w * 0.2), h - random.uniform(0, h * 0.2)],
        [w - random.uniform(0, w * 0.2), h - random.uniform(0, h * 0.2)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def shear_image(image: np.ndarray) -> np.ndarray:
    """ Apply a random shear transformation to the image """
    h, w = image.shape[:2]
    shear = random.uniform(-0.3, 0.3)
    M = np.float32([[1, shear, 0], [shear, 1, 0]])
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def crop_image(image: np.ndarray) -> np.ndarray:
    """ Randomly crop the image to 80% of its original size """
    h, w = image.shape[:2]
    crop_h, crop_w = int(h * 0.8), int(w * 0.8)
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    return image[top:top + crop_h, left:left + crop_w]


def elastic_distortion(
    image: np.ndarray, alpha: float = 40, sigma: float = 6
) -> np.ndarray:
    """ Apply elastic distortion to the image """
    h, w = image.shape[:2]
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distorted = np.zeros_like(image)
    for i in range(image.shape[2]):
        distorted[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=1, mode='reflect'
        ).reshape(h, w)
    return np.clip(distorted, 0, 255).astype(np.uint8)


def augment_image(image_path: str) -> None:
    """ Load image, apply augmentations, and save outputs """
    image = load_image(image_path)

    augmentations: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "flip": flip_image,
        "rotate": rotate_image,
        "skew": skew_image,
        "shear": shear_image,
        "crop": crop_image,
        "distortion": elastic_distortion
    }

    logger.info("Applying augmentations to: %s", os.path.basename(image_path))
    for name, func in augmentations.items():
        try:
            augmented = func(image)
            save_image(augmented, image_path, name)
        except Exception as e:
            logger.error("Augmentation failed: %s - %s", name, e)


def parse_args() -> argparse.Namespace:
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(
        description="Apply various augmentations to an input image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image file."
    )
    return parser.parse_args()


def main() -> None:
    """ Main function to execute the script """
    args = parse_args()

    if not os.path.isfile(args.image_path):
        logger.error("File not found: %s", args.image_path)
        return

    try:
        augment_image(args.image_path)
        logger.info("All augmentations applied successfully.")
    except Exception as e:
        logger.exception("Failed to process image: %s", e)


if __name__ == "__main__":
    main()

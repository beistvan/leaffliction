# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#   Augmentation.py                                    :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#   By: iberegsz <iberegsz@student.42vienna.com>   +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#   Created: 2025/03/31 12:36:41 by iberegsz          #+#    #+#              #
#   Updated: 2025/04/02 23:28:17 by iberegsz         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import Image
import numpy as np
import random
import sys
import cv2
import os


def load_image(image_path):
    """Load an image using OpenCV in RGB format.

    Args:
        image_path (str): Path to the image file.

    Raises:
        FileNotFoundError: If the image file cannot be found or read.
        ImportError: If OpenCV is not installed correctly.

    Returns:
        np.ndarray: The loaded image in RGB format.
    """
    try:
        if not hasattr(cv2, 'imread') or not hasattr(cv2, 'cvtColor'):
            raise ImportError(
                "OpenCV (cv2) is not installed correctly or there is a naming "
                "conflict."
            )

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error in load_image: {e}")
        sys.exit(1)


def save_image(image, original_path, suffix):
    """Save the image with the original filename and the augmentation type.

    Args:
        image (np.ndarray): The image to save.
        original_path (str): Path to the original image file.
        suffix (str): Suffix to append to the filename.

    Raises:
        Exception: If there is an error during saving the image.

    Returns:
        None
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.basename(original_path)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{suffix}{ext}"
        save_path = os.path.join(script_dir, new_filename)
        Image.fromarray(image).save(save_path)
    except Exception as e:
        print(f"Error in save_image: {e}")
        sys.exit(1)


def flip_image(image):
    """Flip the image horizontally.

    Args:
        image (np.ndarray): The image to flip.

    Raises:
        Exception: If there is an error during flipping the image.

    Returns:
        np.ndarray: The flipped image.
    """
    try:
        return cv2.flip(image, 1)
    except Exception as e:
        print(f"Error in flip_image: {e}")
        raise


def rotate_image(image):
    """Rotate the image by a random angle between -30 and 30 degrees.

    Args:
        image (np.ndarray): The image to rotate.

    Raises:
        Exception: If there is an error during rotation.

    Returns:
        np.ndarray: The rotated image.
    """
    try:
        angle = random.uniform(-30, 30)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    except Exception as e:
        print(f"Error in rotate_image: {e}")
        raise


def skew_image(image):
    """Apply a random perspective skew.

    Args:
        image (np.ndarray): The image to skew.

    Raises:
        Exception: If there is an error during skewing.

    Returns:
        np.ndarray: The skewed image.
    """
    try:
        h, w = image.shape[:2]
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst_pts = np.float32([
            [random.uniform(0, w * 0.2), random.uniform(0, h * 0.2)],
            [w - random.uniform(0, w * 0.2), random.uniform(0, h * 0.2)],
            [random.uniform(0, w * 0.2), h - random.uniform(0, h * 0.2)],
            [w - random.uniform(0, w * 0.2), h - random.uniform(0, h * 0.2)]
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(
            image, M, (w, h), borderMode=cv2.BORDER_REFLECT
        )
    except Exception as e:
        print(f"Error in skew_image: {e}")
        raise


def shear_image(image):
    """Apply a shear transformation.

    Args:
        image (np.ndarray): The image to shear.

    Raises:
        Exception: If there is an error during shearing.

    Returns:
        np.ndarray: The sheared image."""
    try:
        h, w = image.shape[:2]
        shear_factor = random.uniform(-0.3, 0.3)
        M = np.float32([[1, shear_factor, 0], [shear_factor, 1, 0]])
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    except Exception as e:
        print(f"Error in shear_image: {e}")
        raise


def crop_image(image):
    """Randomly crop a part of the image.

    Args:
        image (np.ndarray): The image to crop.

    Raises:
        Exception: If there is an error during cropping.

    Returns:
        np.ndarray: The cropped image.
    """
    try:
        h, w = image.shape[:2]
        crop_x = int(w * 0.8)
        crop_y = int(h * 0.8)
        start_x = random.randint(0, w - crop_x)
        start_y = random.randint(0, h - crop_y)
        return image[start_y:start_y + crop_y, start_x:start_x + crop_x]
    except Exception as e:
        print(f"Error in crop_image: {e}")
        raise


def elastic_distortion(image, alpha=40, sigma=6):
    """Apply elastic transformation (distortion).

    Args:
        image (np.ndarray): The image to distort.
        alpha (float): Scaling factor for the distortion.
        sigma (float): Standard deviation for the Gaussian filter.

    Raises:
        Exception: If there is an error during distortion.

    Returns:
        np.ndarray: The distorted image.
    """
    try:
        h, w = image.shape[:2]
        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.array([y + dy, x + dx])

        distorted_channels = []
        for c in range(image.shape[2]):
            distorted_channel = map_coordinates(
                image[:, :, c],
                indices,
                order=1,
                mode='reflect',
                prefilter=False
            )
            distorted_channels.append(distorted_channel)

        distorted = np.stack(distorted_channels, axis=-1)
        return np.clip(distorted, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error in elastic_distortion: {e}")
        raise


def augment_image(image_path):
    """Apply all augmentations and save images.

    Args:
        image_path (str): Path to the image file.

    Raises:
        Exception: If there is an error during augmentation.

    Returns:
        None
    """
    try:
        image = load_image(image_path)

        augmentations = {
            "Flip": flip_image,
            "Rotate": rotate_image,
            "Skew": skew_image,
            "Shear": shear_image,
            "Crop": crop_image,
            "Distortion": elastic_distortion
        }

        for name, func in augmentations.items():
            try:
                augmented_image = func(image)
                save_image(augmented_image, image_path, name)
            except Exception as e:
                print(f"Error applying augmentation {name}: {e}")
    except Exception as e:
        print(f"Error in augment_image: {e}")
        sys.exit(1)


def main():
    """Main function to apply augmentations to an image."

    Args:
        None

    Raises:
        SystemExit: If the script is not run with the correct arguments.
        FileNotFoundError: If the image file cannot be found or read.
        ImportError: If OpenCV is not installed correctly.

    Returns:
        None
    """
    try:
        if len(sys.argv) < 2:
            print(f"Usage: {sys.argv[0]} <path_to_image>")
            sys.exit(1)

        image_path = sys.argv[1]
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"{image_path} is not a valid file.")

        augment_image(image_path)
        print("Augmented images saved successfully.")
    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

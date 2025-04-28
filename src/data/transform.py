"""
This script applies a series of transformations to either a single image
or all images in a directory. It supports both GUI display and headless
saving of outputs. Transformations include:
- Gaussian blur
- Contrast adjustment
- HSV masking
- Region of interest (ROI) detection
- Object analysis (area and perimeter)
- Pseudo-landmark detection
- RGB color histogram plotting
"""
import argparse
import logging
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from transformations import get_transformations
from intensity import plot_color_histogram
from plantcv import plantcv as pcv


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_image(image_path: str) -> np.ndarray:
    """ Load an image in RGB format from a given file path """
    image, _, _ = pcv.readimage(image_path, mode='RGB')
    return image


def save_image(image: np.ndarray, save_path: str) -> None:
    """ Save an RGB image to disk in BGR format (OpenCV default)"""
    # cv2.imwrite(save_path, image
    pcv.print_image(image, save_path)
    logger.info("Saved: %s", save_path)


def display_transformations(image: np.ndarray) -> None:
    """ Display all transformed images and color histogram in a grid """
    fig = plt.figure(figsize=(12, 12))

    for i, (name, function) in enumerate(get_transformations().items()):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(function(image.copy()), cmap='gray')
        ax.set_title(f"Figure {i+1}: {name.replace('_', ' ')}")
        ax.axis('off')

    ax = fig.add_subplot(3, 3, 8)
    plot_color_histogram(image, ax)
    ax.axis('on')

    plt.tight_layout()
    plt.show()


def save_transformations(image_path: str, dst_dir: str, hist: bool) -> None:
    """ Save all transformations and histogram of an image """
    image = load_image(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(dst_dir, exist_ok=True)

    for name, function in get_transformations().items():
        save_path = os.path.join(dst_dir, f"{base_name}_{name}.jpg")
        save_image(function(image), save_path)
        logger.info("Saved: %s", save_path)

    if hist:
        hist_path = os.path.join(dst_dir, f"{base_name}_Color_histogram.png")
        fig, ax = plt.subplots()
        plot_color_histogram(image, ax)
        plt.savefig(hist_path)
        plt.close(fig)
        logger.info("Saved: %s", hist_path)


def process_directory(src_dir: str, dst_dir: str) -> None:
    """ Apply transformations to all supported images in a directory """
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    if not os.path.isdir(src_dir):
        sys.exit(f"Error: {src_dir} is not a valid directory.")

    for entry in os.scandir(src_dir):
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in exts:
            save_transformations(entry.path, dst_dir, False)


def run_single_image(path: str, headless: bool) -> None:
    """ Process a single image in headless or GUI mode """
    if not os.path.isfile(path):
        sys.exit(f"Error: {path} is not a valid file.")

    image = load_image(path)

    if headless:
        os.makedirs("output", exist_ok=True)
        save_transformations(path, "output", True)

    display_transformations(image)


def parse_args() -> argparse.Namespace:
    """ Parse and validate command-line arguments """
    parser = argparse.ArgumentParser(description="Image transformation tool")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-path", help="Path to a single image")
    group.add_argument("-src", help="Directory containing images")

    parser.add_argument(
        "-dst", help="Destination directory (required with -src)"
    )
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Disable GUI display"
    )

    args = parser.parse_args()

    if args.src and not args.dst:
        parser.error("-dst is required when using -src")

    return args


def main() -> None:
    """ Entry point of the script """
    args = parse_args()

    if args.path:
        run_single_image(args.path, args.headless)
    else:
        process_directory(args.src, args.dst)


if __name__ == "__main__":
    main()

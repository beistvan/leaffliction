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

import os
import sys
import cv2
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_image(image_path: str) -> np.ndarray:
    """ Load an image in RGB format from a given file path """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        sys.exit(f"Error: Could not read image {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, save_path: str) -> None:
    """ Save an RGB image to disk in BGR format (OpenCV default)"""
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)


def transformation_original(image: np.ndarray) -> np.ndarray:
    """ Return a copy of the original image """
    return image.copy()


def transformation_gaussian(image: np.ndarray) -> np.ndarray:
    """ Apply Gaussian blur to the image """
    return cv2.GaussianBlur(image, (7, 7), 0)


def transformation_contrast(
    image: np.ndarray, alpha=1.5, beta=0
) -> np.ndarray:
    """ Adjust image contrast using scaling (alpha) and shifting (beta) """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def transformation_mask(image: np.ndarray) -> np.ndarray:
    """Apply HSV masking to isolate green-ish areas in the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def transformation_roi(image: np.ndarray) -> np.ndarray:
    """ Draw bounding rectangles around masked regions (ROI detection) """
    mask = cv2.cvtColor(transformation_mask(image), cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    image_roi = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image_roi


def transformation_analyze(image: np.ndarray) -> np.ndarray:
    """ Draw the largest contour and annotate its area and perimeter """
    mask = cv2.cvtColor(transformation_mask(image), cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    image_analyze = image.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        cv2.drawContours(image_analyze, [largest], -1, (0, 255, 0), 2)
        cv2.putText(image_analyze, f"Area: {int(area)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image_analyze, f"Perimeter: {int(perimeter)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return image_analyze


def transformation_pseudolandmarks(image: np.ndarray) -> np.ndarray:
    """ Detect pseudo-landmarks using Shi-Tomasi corner detection """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=50,
        qualityLevel=0.01,
        minDistance=10
    )
    image_landmarks = image.copy()
    if corners is not None:
        corners = np.intp(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(image_landmarks, (x, y), 3, (255, 0, 255), -1)
    return image_landmarks


def plot_color_histogram(image: np.ndarray, ax) -> None:
    """ Plot the RGB color histogram of the image """
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
        ax.set_xlim([0, 256])
    ax.set_title("Color Histogram")


def get_transformations(image: np.ndarray) -> Dict[str, np.ndarray]:
    """ Apply and return a dictionary of all transformations """
    return {
        "Original": transformation_original(image),
        "Gaussian_blur": transformation_gaussian(image),
        "Contrast_adjusted": transformation_contrast(image),
        "Mask": transformation_mask(image),
        "ROI_objects": transformation_roi(image),
        "Analyze_object": transformation_analyze(image),
        "Pseudolandmarks": transformation_pseudolandmarks(image)
    }


def display_transformations(
    image: np.ndarray, transformations: Dict[str, np.ndarray]
) -> None:
    """ Display all transformed images and color histogram in a grid """
    titles = list(f"Figure IV.{i+1}: {name.replace('_', ' ')}"
                  for i, name in enumerate(transformations.keys()))
    titles.append("Figure IV.8: Color histogram")

    fig = plt.figure(figsize=(12, 12))
    for i, (_, img) in enumerate(transformations.items()):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(img)
        ax.set_title(titles[i])
        ax.axis('off')

    ax = fig.add_subplot(3, 3, 8)
    plot_color_histogram(image, ax)
    ax.axis('on')

    plt.tight_layout()
    plt.show()


def save_transformations(image_path: str, dst_dir: str) -> None:
    """ Save all transformations and histogram of an image to disk """
    image = load_image(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(dst_dir, exist_ok=True)

    transformations = get_transformations(image)

    for name, img in transformations.items():
        save_path = os.path.join(dst_dir, f"{base_name}_{name}.jpg")
        save_image(img, save_path)
        logger.info("Saved: %s", save_path)

    hist_path = os.path.join(dst_dir, f"{base_name}_Color_histogram.png")
    fig, ax = plt.subplots()
    plot_color_histogram(image, ax)
    plt.savefig(hist_path)
    plt.close(fig)
    logger.info("Saved: %s", hist_path)



def process_directory(src_dir: str, dst_dir: str) -> None:
    """ Apply transformations to all supported images in a directory """
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for entry in os.scandir(src_dir):
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in exts:
            save_transformations(entry.path, dst_dir)


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


def run_single_image(path: str, dst: Optional[str], headless: bool) -> None:
    """ Process a single image in headless or GUI mode """
    if not os.path.isfile(path):
        sys.exit(f"Error: {path} is not a valid file.")

    image = load_image(path)
    transformations = get_transformations(image)

    if headless:
        dst = dst or "output"
        os.makedirs(dst, exist_ok=True)
        save_transformations(path, dst)
    else:
        display_transformations(image, transformations)


def main() -> None:
    """ Entry point of the script """
    args = parse_args()

    if args.src:
        if not os.path.isdir(args.src):
            sys.exit(f"Error: {args.src} is not a valid directory.")
        process_directory(args.src, args.dst)
    else:
        run_single_image(args.path, args.dst, args.headless)


if __name__ == "__main__":
    main()

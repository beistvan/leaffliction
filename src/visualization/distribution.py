"""
This script analyzes the distribution of images across subdirectories.
It generates visualizations to represent the distribution of images
for each subdirectory.

Usage:
- Provide path to the top-level directory containing subdirectories with images
- Use the `--save` to save the plots instead of displaying them interactively
"""
import argparse
import os
import logging
from typing import Dict

import matplotlib.pyplot as plt


VALID_IMAGE_EXTENSIONS = {
    '.jpg',
    '.jpeg',
    '.png',
    '.gif',
    '.bmp',
    '.tif',
    '.tiff'
}

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def is_image_file(filename: str) -> bool:
    """ Check if a file is an image based on its extension """
    return os.path.splitext(filename.lower())[1] in VALID_IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(
        usage="%(prog)s -data <path_to_directory> --save",
        description="Visualize image distribution in subdirectories"
    )
    parser.add_argument(
        "-data", "--data-dir",
        type=str,
        required=True,
        help="Path to the top-level directory containing image subdirectories."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the plot as an image file instead of displaying it."
    )
    return parser.parse_args()


def analyze_image_distribution(directory: str) -> Dict[str, int]:
    """ Analyze image distribution across subdirectories """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"{directory} is not a valid directory.")

    distribution = {}
    for entry in os.scandir(directory):
        if entry.is_dir():
            subdir = os.path.basename(entry.path)
            try:
                image_count = sum(
                    1 for _, _, files in os.walk(entry.path)
                    for f in files if is_image_file(f)
                )
                if image_count > 0:
                    distribution[subdir] = image_count
                    logger.debug("Found %d images in %s", image_count, subdir)
                else:
                    logger.warning("No images found in subdir %s", subdir)
            except Exception as e:
                logger.error("Error processing directory %s %s", entry.path, e)
    return distribution


def plot_distribution(
    distribution: Dict[str, int], dataset_name: str, save: bool = False
) -> None:
    """ Plot and optionally save distribution of images """
    if not distribution:
        logger.warning("No image data to plot.")
        return

    labels, counts = zip(
        *sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    )
    cmap = plt.get_cmap("tab20c")
    colors = [cmap(i / len(labels)) for i in range(len(labels))]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].pie(
        counts,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors
    )
    axs[0].set_title(f"{dataset_name} - Pie Chart")

    axs[1].bar(labels, counts, color=colors)
    axs[1].set_title(f"{dataset_name} - Bar Chart")
    axs[1].tick_params(axis='x', rotation=45)
    fig.tight_layout()

    if save:
        output_file = f"{dataset_name}_distribution.png"
        fig.savefig(output_file, bbox_inches='tight')
        logger.info("Plot saved to: %s", output_file)
    else:
        plt.show()

    plt.close(fig)


def main():
    """ Main function to execute the script """
    args = parse_args()

    dataset_abspath = os.path.abspath(args.data_dir.rstrip("/\\"))
    dataset_name = os.path.basename(dataset_abspath)

    try:
        logger.info("Analyzing dataset: %s", dataset_name)
        distribution = analyze_image_distribution(args.data_dir)
        plot_distribution(distribution, dataset_name, save=args.save)
    except NotADirectoryError as e:
        logger.error("Directory error: %s", e)
    except Exception as e:
        logger.exception("Unexpected error occurred during execution %s", e)


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import subprocess
import logging
from typing import Dict
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balance dataset by augmenting images"
    )
    parser.add_argument(
        "-src",
        type=str,
        required=True,
        help="Directory containing the input images"
    )
    parser.add_argument(
        "-dst",
        type=str,
        required=True,
        help="Directory to save the balanced images"
    )
    parser.add_argument(
        "-aug",
        type=str,
        required=True,
        help="Path to the augmentation script"
    )
    return parser.parse_args()


def get_distributions(src_dir: str) -> Dict[str, int]:
    """ Get the distribution of images in the source directory """
    distributions = defaultdict(int)
    for root, _, files in os.walk(src_dir):
        if not files:
            continue

        label = root.split(os.sep)[-1]
        for file in files:
            if file.endswith('.JPG'):
                distributions[label] += 1

        logger.info(f"Counted {distributions[label]} images in {label} class")
    return distributions


def get_to_augument(distributions: Dict[str, int]) -> Dict[str, int]:
    """ Get the number of images to balance each class """
    max_count = max(distributions.values())
    return {label: max_count - count for label, count in distributions.items()}


def copy_images(src: str, dst: str):
    """ Copy images from source to destination directory """
    for label in os.listdir(src):
        src_label_dir = os.path.join(src, label)
        dst_label_dir = os.path.join(dst, label)
        os.makedirs(dst_label_dir, exist_ok=True)

        for file in os.listdir(src_label_dir):
            if file.endswith('.JPG'):
                src_path = os.path.join(src_label_dir, file)
                dst_path = os.path.join(dst_label_dir, file)
                with open(src_path, 'rb') as src_file:
                    with open(dst_path, 'wb') as dst_file:
                        dst_file.write(src_file.read())


def delete_random_file(label_dir: str):
    """ Delete a random file from the label directory """
    files = [f for f in os.listdir(label_dir) if f.endswith('.JPG')]
    if not files:
        return

    file_to_delete = os.path.join(label_dir, files[0])
    os.remove(file_to_delete)


def create_augumented_images(
    src: str,
    to_augument: Dict[str, int],
    script: str
) -> None:
    """ Create augmented images to balance the dataset """
    for label, count in to_augument.items():
        if count <= 0:
            continue

        logger.info(f"Start processing {count} images in {label} class")

        label_dir = os.path.join(src, label)
        for file in os.listdir(label_dir):
            if file.endswith('.JPG'):
                subprocess.run([
                    sys.executable,
                    script,
                    os.path.join(label_dir, file),
                    '--silent'
                ])

            to_augument[label] -= 6
            if to_augument[label] <= 0:
                break

        for _ in range(abs(to_augument[label])):
            delete_random_file(label_dir)


def main():
    args = parse_args()
    src_dir = args.src
    dst_dir = args.dst

    if not os.path.exists(src_dir):
        logger.error(f"Source directory {src_dir} does not exist.")
        return

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    if src_dir == dst_dir:
        logger.error("Source and destination directories must be different.")
        return

    if not os.path.exists(args.aug):
        logger.error(f"Augmentation script {args.aug} does not exist.")
        return

    try:
        distributions = get_distributions(src_dir)
        to_augument = get_to_augument(distributions)
        copy_images(src_dir, dst_dir)
        create_augumented_images(dst_dir, to_augument, args.aug)
        logger.info("Dataset balancing completed.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

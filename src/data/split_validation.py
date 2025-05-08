import argparse
import os
import logging
import random

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args() -> argparse.Namespace:
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(
        description="Creates validation dataset from the original dataset."
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the dataset root directory"
    )
    parser.add_argument(
        "-s",
        "--split",
        type=int,
        default=20,
        help="Amount of images to be used for validation"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="validation_dataset",
        help="Directory to save the validation dataset"
    )
    return parser.parse_args()


def main() -> None:
    """ Main function to execute the script """
    args = parse_args()

    try:
        folders = os.listdir(args.dataset_dir)
        logger.info(f"Found {len(folders)} folders in the dataset directory.")
        if not folders:
            logger.error("No folders found in the dataset directory.")
            return

        for folder in folders:
            src_folder = os.path.join(args.dataset_dir, folder)
            images = os.listdir(src_folder)

            assert len(images) >= args.split, f'Not enough images in {folder}'

            for _ in range(args.split):
                image = images.pop(random.randint(0, len(images) - 1))
                src_image = os.path.join(src_folder, image)
                dst_image = os.path.join(args.output_dir, folder, image)
                os.makedirs(os.path.dirname(dst_image), exist_ok=True)
                os.rename(src_image, dst_image)

            logger.info(f"Successfully processed {folder} folder.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

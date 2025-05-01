import sys
import argparse
import os
from tqdm import tqdm
import logging
from typing import NoReturn
from transformations import ImageTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class Parameters:
    """ Stores parameters for the transformation process """
    def __init__(
            self,
            image_path,
            debug="print",
            writeimg=True,
            outdir=".",
            display=False):
        self.image = image_path
        self.debug = debug
        self.writeimg = writeimg
        self.outdir = outdir
        self.display = display
        os.makedirs(self.outdir, exist_ok=True)


def transform_image(params: Parameters, display=False) -> NoReturn:
    """ Generate image transformations for a single image """
    transformer = ImageTransformer(
        params.image,
        params.debug,
        params.writeimg,
        params.outdir,
        params.display
    )
    transformer.apply_gaussian_blur()
    transformer.apply_mask()
    transformer.compute_roi()
    transformer.analyze_objects()
    transformer.generate_pseudolandmarks()
    if display:
        transformer.display_results()


def transform_batch(src: str, dst: str) -> NoReturn:
    """ Generate image transformations for a batch of images """
    logger.info(f"Batch processing images from {src} to {dst}")
    logger.info(f"Found {len(os.listdir(src))} pictures")

    for root, _, files in os.walk(src):
        images = [file for file in files if file.endswith(".JPG")]

        if not images:
            continue

        for file in tqdm(images):
            params = Parameters(
                os.path.join(root, file),
                debug="print",
                writeimg=True,
                outdir=os.path.join(dst, root),
            )
            transform_image(params, False)


def extract_name(path: str) -> str:
    """ Returns filename without extension """
    base = os.path.basename(os.path.normpath(path))
    stem, _ = os.path.splitext(base)
    return stem


def parse_args() -> argparse.Namespace:
    """ Parses command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("img", nargs="?", type=str, help="Image to process")
    parser.add_argument(
        "-src",
        type=str,
        help="Source dir with images")
    parser.add_argument(
        "-dst",
        type=str,
        help="Destination dir for processed images")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> NoReturn:
    """ Validates command line arguments """
    if (not args.src or not args.dst) and not args.img:
        print("usage:")
        print(f"{os.path.basename(__file__)} <img>")
        print(f"{os.path.basename(__file__)} -src <dir> -dst <dir> [--hist]")
        sys.exit(1)

    if args.img:
        if not os.path.isfile(args.img):
            logger.error(f"Image file '{args.img}' does not exist")
            sys.exit(1)
        if not args.img.endswith(".JPG"):
            logger.error("The provided file is not a JPG image")
            sys.exit(1)
        return

    if not args.src or not args.dst:
        logger.error("Source and image arguments are required")
        sys.exit(1)

    if not os.path.isdir(args.src):
        logger.error(f"Source directory '{args.src}' does not exist")
        sys.exit(1)

    if not os.path.isdir(args.dst):
        logger.info(f"Destination directory '{args.dst}' does not exist")
        os.makedirs(args.dst)
        logger.info(f"Destination directory '{args.dst}' created")


def main() -> NoReturn:
    """ Main function to run the image transformation process """
    args = parse_args()
    validate_args(args)

    if args.img:
        params = Parameters(args.img, outdir="./images", display=True)
        transform_image(params, True)
        logger.info('Image transformation completed.')
        logger.info(f'Results saved in {params.outdir}')
    else:
        transform_batch(args.src, args.dst)
        logger.info('Batch image transformation completed.')
        logger.info(f'Results saved in {args.dst}')


if __name__ == "__main__":
    main()

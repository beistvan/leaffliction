import argparse
import logging
import pickle
import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from train import MODEL_FILENAME, CLASSES_FILENAME

matplotlib.use('TkAgg')

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RESULT_FILE = 'results.txt'


def parse_args() -> argparse.Namespace:
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(
        description="Predict leaf desease using a trained model"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-image",
        help="Path to the input image."
    )
    group.add_argument(
        "-dir",
        help="Path to the directory containing images."
    )
    parser.add_argument(
        "model_zip",
        help="Path to the zip file containing the trained model and classes.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        default=False,
        help="Save the prediction results to a file."
    )
    return parser.parse_args()


def predict_image(model: tf.keras.Model, img_path: str) -> tuple[int, float]:
    """ Predict the class of an image using the trained model """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]
    return class_idx, confidence


def unzip(zip_path: str) -> tuple[tf.keras.Model, list[str]]:
    """ Unzips the archive and loads the model and classes """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extract(MODEL_FILENAME, path=".")
        zf.extract(CLASSES_FILENAME, path=".")
    model = tf.keras.models.load_model(MODEL_FILENAME)
    with open(CLASSES_FILENAME, 'rb') as f:
        classes = pickle.load(f)
    return model, classes


def plot_results(img_path: str, confidence: float, predicted: str):
    """ Plot the original and resized images with predictions """
    img = image.load_img(img_path)

    plt.figure(num='Prediction')
    plt.imshow(img)
    plt.text(
        img.width / 2,
        img.height - 10,
        f"Pred: {predicted}\nConf: {confidence:.2f}",
        fontsize=12,
        ha='center',
        va='center',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
    )
    plt.title(f"Image: {img_path}")
    plt.tight_layout()
    plt.axis('off')
    plt.show()


def validate_arguments(args: argparse.Namespace):
    """ Validate command line arguments """
    if not os.path.isfile(args.model_zip):
        logger.error(f"Error: {args.model_zip} does not exist.")

    if args.image:
        if not os.path.isfile(args.image):
            logger.error(f"Error: {args.image} does not exist.")
        return

    if not os.path.isdir(args.dir):
        logger.error(f"Error: {args.dir} does not exist.")
        return


def extract_images(image_path: str, dir_path: str) -> list[str]:
    """ Extract images from the specified path """
    if image_path:
        return [image_path]

    images = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith('jpg'):
                images.append(os.path.join(root, file))
    return images


def store_predictions(predictions: tuple[int, float, str]):
    """ Stores predictions in result file """
    total = len(predictions)
    successful = 0

    with open(RESULT_FILE, 'w') as f:
        for path, _, prediction in predictions:
            f.write(f'{path} - {prediction}\n')
            if prediction.lower() in path.lower():
                successful += 1
        f.write(f"Expected accuracy {successful}/{total} or {successful / total}\n")


def main():
    """ Main function to load the model and predict the class of the image """
    args = parse_args()
    validate_arguments(args)
    images = extract_images(args.image, args.dir)
    logger.info(f"Found {len(images)} images to predict.")

    model, classes = unzip(args.model_zip)
    logger.info(f"Model loaded from {args.model_zip}")

    predictions = []
    for img_path in images:
        class_idx, confidence = predict_image(model, img_path)
        predictions.append((img_path, confidence, classes[class_idx]))

    if args.silent:
        store_predictions(predictions)
    else:
        for img, conf, pred in predictions:
            logger.info(f"Predicting {img_path}")
            plot_results(img, conf, pred)

    logger.info("Done")


if __name__ == "__main__":
    main()

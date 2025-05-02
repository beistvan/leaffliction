import argparse
import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import logging
import pypickle
from train import MODEL_FILENAME, CLASSES_FILENAME

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args() -> argparse.Namespace:
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(
        description="Predict leaf desease using a trained model"
    )

    parser.add_argument(
        "image_path",
        help="Path to the input image."
    )
    parser.add_argument(
        "-tmz",
        required=True,
        help="Path to the zip file containing the trained model and classes.",
    )
    parser.add_argument(
        "--save",
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
    classes = pypickle.load(CLASSES_FILENAME)
    return model, classes


def plot_results(img_path: str, confidence: float, predicted: str, save: bool):
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
    if save:
        plt.savefig(f"prediction_{os.path.basename(img_path)}")
    plt.show()


def main():
    """ Main function to load the model and predict the class of the image """
    args = parse_args()

    if not os.path.isfile(args.image_path):
        logger.error(f"Error: {args.image_path} does not exist.")
        return

    if not os.path.isfile(args.tmz):
        logger.error(f"Error: {args.tmz} does not exist.")
        return

    model, classes = unzip(args.tmz)
    logger.info(f"Model loaded from {args.tmz}")

    class_idx, confidence = predict_image(model, args.image_path)
    plot_results(args.image_path, confidence, classes[class_idx], args.save)
    logger.info("Done")


if __name__ == "__main__":
    main()

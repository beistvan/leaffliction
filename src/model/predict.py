# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#   predict.py                                         :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#   By: iberegsz <iberegsz@student.42vienna.com>   +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#   Created: 2025/04/01 11:43:25 by iberegsz          #+#    #+#              #
#   Updated: 2025/04/14 12:00:39 by iberegsz         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import argparse
import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Predict leaf disease from a single image
using a trained model.""")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument(
        "--model_zip",
        default="trained_model_and_augmented.zip",
        help="Zip file containing the trained model.")
    parser.add_argument(
        "--model_file",
        default="trained_leaf_disease_model.h5",
        help="H5 model file inside the zip.")
    return parser.parse_args()


def load_model_from_zip(zip_path, model_filename):
    """
    Extracts the .h5 file from the zip, loads it with Keras,
    then returns the model.
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extract(model_filename, path=".")
    model = tf.keras.models.load_model(model_filename)
    return model


def predict_image(model, img_path):
    """
    Preprocess the image, run model prediction, return predicted
    class and probability.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]
    return class_idx, confidence


def main():
    """
    Main function to parse arguments, load the model,
    predict the class of the image, and display results.
    """
    args = parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: {args.image_path} is not a valid file.")
        return

    if not os.path.isfile(args.model_zip):
        print(f"Error: {args.model_zip} does not exist.")
        return

    model = load_model_from_zip(args.model_zip, args.model_file)
    print("Model loaded successfully.")

    class_idx, confidence = predict_image(model, args.image_path)

    class_names = [
        "apple_apple_scab",
        "apple_black_rot",
        "apple_cedar_apple_rust",
        "apple_healthy"]

    if class_idx < len(class_names):
        predicted_class = class_names[class_idx]
    else:
        predicted_class = "Unknown"

    img_original = image.load_img(args.image_path)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    resized_img = image.load_img(args.image_path, target_size=(224, 224))
    axes[1].imshow(resized_img)
    axes[1].set_title(
        f"""Transformed (Resized)
Pred: {predicted_class}
Conf: {confidence:.2f}"""
    )
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

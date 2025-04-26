"""
Train a leaf disease classification model using TensorFlow and Keras.
This script includes data augmentation and saves the trained model
and augmented images in a zip file.
"""
import argparse
import zipfile
import shutil
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a leaf disease classification model."
    )

    parser.add_argument("--data", help="Path to the dataset root directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs.")
    parser.add_argument("--output", default="trained_model_and_augmented.zip", help="""Name of the output zip file containing the model
and augmented images.""")
    return parser.parse_args()


def create_model(num_classes):
    """
    Example CNN model. You can customize or replace
    with a pretrained model (e.g., MobileNet, ResNet, etc.).
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


def main():
    """
    Main function to train the model and save it in a zip file.
    """
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    batch_size = args.batch_size

    output_zip = args.output_zip

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        directory=data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    val_generator = val_datagen.flow_from_directory(
        directory=data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    num_classes = len(train_generator.class_indices)
    print(f"Found {num_classes} classes: {train_generator.class_indices}")

    model = create_model(num_classes)
    model.summary()

    val_loss, val_acc = model.evaluate(val_generator)
    print(f"Validation accuracy: {val_acc:.2f}")

    model.save("trained_leaf_disease_model.h5")
    print("Model saved as trained_leaf_disease_model.h5")

    augmented_folder = "augmented_images"
    if os.path.exists(augmented_folder):
        shutil.rmtree(augmented_folder)
    os.makedirs(augmented_folder, exist_ok=True)

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write("trained_leaf_disease_model.h5")

        for root, dirs, files in os.walk(augmented_folder):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, start=".")
                zf.write(full_path, arcname=relative_path)

    print(f"Zipped model and augmented images into {output_zip}")


if __name__ == "__main__":
    main()

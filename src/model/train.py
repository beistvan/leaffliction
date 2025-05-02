from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import argparse
import zipfile
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args() -> argparse.Namespace:
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(
        description="Train a leaf disease classification model."
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to the dataset root directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--output_zip",
        default="trained_model.zip",
        help="Name of the output zip with the model and augmented images"
    )
    return parser.parse_args()


def create_model(num_classes: int) -> models.Sequential:
    """ Create a simple CNN model for leaf disease classification """
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
        metrics=['accuracy']
    )
    return model


def setup_data_generators(
            data_dir: str, batch_size: int
        ) -> tuple[ImageDataGenerator, ImageDataGenerator]:
    """ Set up data generators for training and validation """
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

    return train_generator, val_generator


def save_trainings(
            model: models.Sequential,
            output_zip: str
        ) -> None:
    """ Save the trained model and augmented images in a zip file """
    model.save("trained_model.h5")
    logger.info("Model saved as trained_model.h5")

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write("trained_model.h5")

        for root, _, files in os.walk("augmented_images"):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, start=".")
                zf.write(full_path, arcname=relative_path)

    logger.info(f"Model and augmented images saved in {output_zip}")


def main():
    """ Main function to train the model and save it in a zip file """
    args = parse_args()

    train_generator, val_generator = setup_data_generators(
        os.path.abspath(args.data_dir),
        args.batch_size
    )
    logger.info("Start training")

    model = create_model(len(train_generator.class_indices))

    model.summary()
    model.fit(
        train_generator,
        validation_data=val_generator,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1
    )

    val_loss, val_acc = model.evaluate(val_generator)
    logger.info(f"Validation loss {val_loss}, accuracy {val_acc}")

    save_trainings(model, args.output_zip)
    logger.info("Training and saving completed.")


if __name__ == "__main__":
    main()

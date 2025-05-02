# Leaf Disease Classification Project

## Project Description

This project focuses on **leaf disease classification** using deep learning. The model is trained on an augmented dataset of leaf images, with preprocessing steps such as dataset balancing, image transformation, and augmentation. The goal is to predict the disease category of a given leaf image.

## Setup and Run Instructions

### Prerequisites

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

### 1. Install Dataset

First, download and extract the dataset:
```bash
wget https://cdn.intra.42.fr/document/document/17547/leaves.zip
unzip leaves.zip
```

### 2. Check Image Distribution

Run the following command to visualize the current image distribution:
```bash
python src/visualization/distribution.py -data images
```
This will display the distribution of images across different classes, allowing you to check if the dataset is balanced.

### 3. Balance the Dataset

To balance the dataset, use the following command. This will automatically augment and balance the dataset, which may take a few minutes:
```bash
python src/data/balance_dataset.py -src images/ -dst balanced_images/ -aug src/data/augment.py
```

### 4. Verify New Distribution

Run the distribution check again to see the updated balanced dataset:
```bash
python src/visualization/distribution.py -data balanced_images/
```

### 5. Image Transformations (Optional)

You can test image transformations (such as rotation, flipping, etc.) to understand how they work with the data. Run this command:
```bash
python src/data/transform.py *pick a random image here*
```
This is mainly for visualization; the model doesn’t require this step.

### 6. Train the Model

Train the model using the augmented and balanced dataset:
```bash
python src/model/train.py --batch_size 30 --epochs 3 --output_zip model.zip augmented_images
```

### 7. Make Predictions

To test the model's predictions, use the following command:
```bash
python src/model/predict.py --model_zip model.zip images/Apple/Apple_rust/image\ \(45\).JPG
```

This will show the model’s prediction for the specified image.

---

Feel free to adjust any details or add further clarifications based on your project's specifics.

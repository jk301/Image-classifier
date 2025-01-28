# Image Classifier using TensorFlow

This project is a deep learning-based image classifier built with TensorFlow. It classifies images into two categories: **humans** and **animals**.

## Features
- Loads and processes images from a directory.
- Preprocesses images for faster training.
- Builds a Convolutional Neural Network (CNN) model using TensorFlow.
- Trains and evaluates the model.
- Makes predictions on new images.

## Dataset
The dataset contains images of humans and animals. These images are stored in subdirectories, each representing a class (`humans` and `animals`).

## Requirements
- Python 3.11 or less
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

## Install Dependencies
To install the required dependencies, run the following command

```plaintext
pip install -r requirements.txt
```
## Running the saved model
To run the saved model use this

```
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Loading it
model = load_model('model/img_classifier.h5')

# Resize the image
img = cv2.imread('path_to_your_image.jpg')
resized_img = cv2.resize(img, (256, 256))

# Normalizing the image for sigmoid activation
img_array = resized_img / 255.0

# Make prediction
prediction = model.predict(np.expand_dims(img_array, axis=0))

# The result
if prediction > 0.5:
    print("It is a human!")
else:
    print("It is an animal!")
```

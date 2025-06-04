# Fashion-MNIST-Nerual-Net

# 🧠 Fashion MNIST Neural Network (NumPy Only)

A from-scratch implementation of a feedforward neural network using **NumPy** to classify images from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). This project does not use machine learning frameworks like TensorFlow or PyTorch—everything is built manually to deepen understanding of neural network mechanics.

## 🔍 Overview

- **Architecture**: 3-layer network (Input → 2 Hidden Layers → Output)
- **Activation Functions**: ReLU for hidden layers, Softmax for output
- **Accuracy**: ~85% on the test set
- **Frameworks**: Only uses `numpy`, `matplotlib`, and `pandas`
- **Dataset**: Fashion MNIST (70,000 grayscale 28x28 images across 10 clothing categories): [https://www.kaggle.com/datasets/zalando-research/fashionmnist](url)

## 🏗️ Features

- Manual forward and backward propagation
- Gradient descent optimization
- Adjustable learning rate, layer sizes, and epochs
- Simple matplotlib visualization for individual test predictions

## How to test against your own image

If you want to test my model against an image of your own, follow these steps:
- Take a picture of the clothing item against a dark background such that the clothing is lighter than the background
- Crop it to a square (resolution and color is handled by my network)
- Upload the image to the Image-Uploads folder & copy the FULL filepath
- Run the photoClassify.py file & input the filepath for the image you want to test!
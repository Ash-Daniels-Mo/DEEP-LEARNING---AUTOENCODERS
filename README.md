# MNIST Autoencoder Experiment

In this practical project, an **autoencoder** was implemented using **PyTorch** and trained on the **MNIST dataset**, which contains images of handwritten digits. The model was primarily trained on **odd digits (1, 3, 5, 7, 9)**, allowing it to learn the underlying structure of these specific digits.

## Purpose

The goal of this experiment is to:

- Observe how well the autoencoder can **reconstruct images it has been trained on**.
- Examine how the model performs on **digits outside the training distribution**.  

Since autoencoders learn by **minimizing reconstruction error**, accurate reconstructions indicate that the model has successfully captured meaningful patterns in the data.

## Project Overview

The project covers the following main components:

1. **Data Preparation** – Loading and preprocessing the MNIST dataset, selecting training digits.
2. **Model Architecture** – Building the encoder and decoder networks.
3. **Training** – Training the autoencoder to minimize reconstruction error.
4. **Reconstruction Results** – Evaluating how well the model reconstructs both seen and unseen digits.

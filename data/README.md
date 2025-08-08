# Data Directory

This directory contains all the datasets used for the Tomato Crop Disease Classification project. It is structured to differentiate between the original, raw data and the processed data ready for model training.

## Structure

* `raw/`: Contains the original, unaltered PlantVillage tomato images as downloaded.

* `processed/`: Will contain preprocessed images (resized, augmented, normalized) split into training, validation, and test sets.

## About Dataset

This dataset consists of images of tomato leaves, categorized by various disease conditions and a healthy state. Each subdirectory within `raw/` represents a specific disease or health status.

## Acknowledgements

This dataset was obtained from [spMohanty's GitHub Repository](https://github.com/spMohanty/PlantVillage-Dataset).

## Inspiration

This dataset was specifically curated for use in a Plant Disease Detection System, aiming to provide a comprehensive collection of images for training and evaluating machine learning models.
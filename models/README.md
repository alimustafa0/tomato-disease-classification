# Models Directory

This directory stores the trained machine learning models and their associated configuration files.

## `model.pth`

This file contains the saved weights of the best-performing trained model.

* **Architecture:** ResNet18 (fine-tuned)
* **Training Dataset:** PlantVillage Tomato Disease Dataset (processed version)
* **Training Date:** [Insert Date of Training, e.g., 2025-08-11]
* **Validation Accuracy (Best):** [Insert Best Validation Accuracy from `src/train.py` output, e.g., 0.9250]
* **Test Accuracy:** [Insert Test Accuracy from `src/evaluate.py` output, e.g., 0.9080]
* **Notes:** This model was trained with data augmentation and early stopping. Refer to `params.yaml` for hyperparameters used during training.

## `model_config.yaml` (Optional, for detailed architecture)

This file (if used) would contain detailed architecture specifics if the model was custom-built or significantly modified beyond standard fine-tuning. For this project, `params.yaml` covers the base model (`resnet18`) and `num_classes`, which might be sufficient. If you later implement a more complex custom architecture, you would detail it here.
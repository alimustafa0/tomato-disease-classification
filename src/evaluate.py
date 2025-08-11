# src/evaluate.py

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import yaml
import sys
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. Configuration Loading ---
print("--- Section 1: Configuration Loading ---")

def load_config(config_path='params.yaml'):
    """
    Loads configuration parameters from a YAML file.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

# Load the main configuration
config = load_config()
train_config = config['train']
data_config = config['data_preprocessing']

# --- 2. Device Configuration ---
print("\n--- Section 2: Device Configuration ---")
device = torch.device(train_config['device'] if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 3. Model Definition (must match training script) ---
print("\n--- Section 3: Model Definition ---")

def build_model(model_name, num_classes, pretrained=True, device="cpu"):
    """
    Builds a pre-trained CNN model and modifies its final layer for classification.
    This function should be identical to the one in train.py to ensure the model
    architecture matches the saved weights.
    """
    if model_name == "resnet18":
        # For evaluation, we don't need to download weights again or freeze layers
        # as we will load the saved state_dict.
        # pretrained=False means it initializes with random weights, then we load our trained ones.
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    print(f"Model '{model_name}' architecture built for {num_classes} output classes.")
    return model.to(device)

# --- 4. Data Loading for Evaluation ---
print("\n--- Section 4: Data Loading for Evaluation ---")

# Define transformations for the test set (no augmentation, only resize and normalize)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
    transforms.Resize(tuple(data_config['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Define paths to your processed data
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'model.pth')

test_dir = os.path.join(PROCESSED_DATA_DIR, 'test')

# Check if test data directory and model file exist
if not os.path.exists(test_dir):
    print(f"Error: Test data directory not found at '{test_dir}'.")
    print("Please ensure '02_data_preprocessing.ipynb' was run successfully.")
    sys.exit(1)

if not os.path.exists(MODEL_SAVE_PATH):
    print(f"Error: Trained model weights not found at '{MODEL_SAVE_PATH}'.")
    print("Please ensure 'src/train.py' was run successfully and saved the model.")
    sys.exit(1)

# Create test dataset
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=4)

class_names = test_dataset.classes
print(f"Loaded test dataset with {len(test_dataset)} images across {len(class_names)} classes.")

# --- 5. Load Model and Weights ---
print("\n--- Section 5: Load Model and Weights ---")

model = build_model(train_config['model_name'], train_config['num_classes'], pretrained=False, device=device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval() # Set model to evaluation mode

print(f"Trained model loaded from: {MODEL_SAVE_PATH}")

# --- 6. Evaluation Function ---
print("\n--- Section 6: Evaluation Function ---")

def evaluate_model(model, test_loader, class_names, device):
    """
    Evaluates the model on the test dataset and prints classification report
    and confusion matrix.
    """
    y_true = []
    y_pred = []

    print("Starting model evaluation on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n--- Classification Report ---")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print("\nEvaluation complete.")

# --- Main Execution Block ---
if __name__ == "__main__":
    evaluate_model(model, test_loader, class_names, device)

# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import yaml
import sys
import copy # For deep copying model state
from tqdm import tqdm # For progress bars

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
        sys.exit(1) # Exit if config is not found
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

# --- 2. Device Configuration ---
print("\n--- Section 2: Device Configuration ---")
# Device will be determined within the main block after config is loaded
# This is a placeholder for clarity.

# --- 3. Data Loading and Transformations ---
print("\n--- Section 3: Data Loading and Transformations ---")

# Data loading and transforms functions will be called within the main block.

# --- 4. Model Definition ---
print("\n--- Section 4: Model Definition ---")

def build_model(model_name, num_classes, pretrained=True, device="cpu"): # Added device arg
    """
    Builds a pre-trained CNN model and modifies its final layer for classification.
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # Freeze all parameters except the final layer if using a pre-trained model
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False
        # Modify the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    # Add more models here if needed (e.g., resnet50, vgg16)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    print(f"Model '{model_name}' built with {num_classes} output classes.")
    return model.to(device) # Move model to the selected device

# --- 5. Loss Function and Optimizer ---
print("\n--- Section 5: Loss Function and Optimizer ---")

# Loss and Optimizer creation will be done in the main block.

# --- 6. Training and Validation Loop ---
print("\n--- Section 6: Training and Validation Loop ---")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs,
                early_stopping_patience, model_save_path, device): # Added device arg
    """
    Trains and validates the model, saving the best performing model.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0 # Counter for early stopping

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Use tqdm for a progress bar during training
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass + optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad(): # No gradient calculation needed for validation
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # --- Early Stopping and Model Saving ---
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0 # Reset counter
            # Save the best model
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # Ensure directory exists
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} with improved validation accuracy: {best_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Validation accuracy did not improve. Early stopping counter: {epochs_no_improve}/{early_stopping_patience}")
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("\nTraining complete.")
    model.load_state_dict(best_model_wts) # Load best model weights
    return model

# --- Main Execution Block ---
if __name__ == "__main__":
    # Load the main configuration
    config = load_config()
    train_config = config['train']
    data_config = config['data_preprocessing']

    # Set up device
    device = torch.device(train_config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize(tuple(data_config['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(tuple(data_config['image_size']), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(tuple(data_config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Define paths to your processed data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')

    train_dir = os.path.join(PROCESSED_DATA_DIR, 'train')
    val_dir = os.path.join(PROCESSED_DATA_DIR, 'val')
    test_dir = os.path.join(PROCESSED_DATA_DIR, 'test') # Though not used in train, good to define

    # Check if processed data directories exist
    if not os.path.exists(train_dir) or not os.path.exists(val_dir) or not os.path.exists(test_dir):
        print(f"Error: Processed data directories not found.")
        print(f"Expected: {train_dir}, {val_dir}, {test_dir}")
        print("Please run '02_data_preprocessing.ipynb' first to generate processed data.")
        sys.exit(1)

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)

    # Create DataLoaders
    # The error suggests num_workers might be an issue on Windows.
    # Set num_workers to 0 or 1 for robustness on Windows if problems persist,
    # though with the __name__ == '__main__' guard, it should ideally work.
    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=4)

    # Get class names from the dataset
    class_names = train_dataset.classes
    print(f"Found {len(class_names)} classes: {class_names}")
    # Verify num_classes in config matches actual classes
    if len(class_names) != train_config['num_classes']:
        print(f"Warning: Config 'num_classes' ({train_config['num_classes']}) does not match actual classes ({len(class_names)}). Adjusting config.")
        train_config['num_classes'] = len(class_names)

    # Build model
    model = build_model(train_config['model_name'], train_config['num_classes'], train_config['pretrained'], device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if train_config['optimizer'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    elif train_config['optimizer'] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=train_config['learning_rate'], momentum=0.9)
    else:
        raise ValueError(f"Optimizer '{train_config['optimizer']}' not supported.")

    print(f"Loss function: {train_config['loss_function']}")
    print(f"Optimizer: {train_config['optimizer']} with learning rate {train_config['learning_rate']}")

    # Define model save path
    MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'model.pth')

    # Start training
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=train_config['epochs'],
        early_stopping_patience=train_config['early_stopping_patience'],
        model_save_path=MODEL_SAVE_PATH,
        device=device
    )

    print(f"\nFinal best model weights saved to: {MODEL_SAVE_PATH}")
    # Note: `best_acc` is local to `train_model`, so we can't print it directly here
    # A workaround would be to return `best_acc` from `train_model` or log it directly.
    # For now, the training progress will show it.

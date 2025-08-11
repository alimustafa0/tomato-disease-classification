# src/data_preprocessing.py

import os
import shutil
import random
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm # For progress bars

# --- 1. Configuration Parameters ---
# These parameters can later be loaded from 'params.yaml' for better configurability.
# For now, they are defined directly here.

# Define the target image size for model input.
# Common sizes are 224x224 or 256x256. Since your raw images are 256x256,
# we can keep it consistent or resize if a model requires a different input.
TARGET_IMAGE_SIZE = (224, 224) # e.g., for common pre-trained models like ResNet

# Define the data split ratios for training, validation, and testing.
TRAIN_SPLIT_RATIO = 0.7 # 70% for training
VAL_SPLIT_RATIO = 0.15 # 15% for validation
TEST_SPLIT_RATIO = 0.15 # 15% for testing (TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT should be 1.0)

# Random seed for reproducibility of splits
RANDOM_SEED = 42

# --- 2. Helper Functions ---

def create_directory_structure(base_dir, classes):
    """
    Creates the necessary directory structure for processed data (train/val/test splits).

    Args:
        base_dir (str): The root directory where 'train', 'val', 'test' folders will be created.
        classes (list): A list of class names (e.g., ['Tomato_healthy', 'Tomato_Bacterial_spot']).
    """
    splits = ['train', 'val', 'test']
    for split in splits:
        split_path = os.path.join(base_dir, split)
        os.makedirs(split_path, exist_ok=True)
        for cls in classes:
            class_path = os.path.join(split_path, cls)
            os.makedirs(class_path, exist_ok=True)
    print(f"Directory structure created at: {base_dir}")

def get_transforms(image_size, data_augmentation=True):
    """
    Defines image transformations including normalization and optional data augmentation.

    Args:
        image_size (tuple): The target (height, width) for resizing.
        data_augmentation (bool): Whether to apply data augmentation for training.

    Returns:
        tuple: (train_transform, val_test_transform)
    """
    # Common normalization values for ImageNet-pretrained models (adjust if using different pre-trained model)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),             # Resize to target size
            transforms.RandomHorizontalFlip(),         # Randomly flip images horizontally
            transforms.RandomVerticalFlip(),           # Randomly flip images vertically
            transforms.RandomRotation(degrees=30),     # Randomly rotate images by up to 30 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Randomly change brightness, contrast, etc.
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)), # Random crop and resize
            transforms.ToTensor(),                     # Convert image to PyTorch Tensor (scales to 0-1)
            normalize                                  # Normalize pixel values
        ])
    else: # No augmentation for validation/test sets
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize
        ])

    # Validation and test transforms should only include resizing and normalization
    val_test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize
    ])

    print("Image transforms defined (with augmentation for training, without for validation/test).")
    return train_transform, val_test_transform

def preprocess_and_split_data(raw_data_dir, processed_data_dir, image_size=TARGET_IMAGE_SIZE,
                              train_ratio=TRAIN_SPLIT_RATIO, val_ratio=VAL_SPLIT_RATIO,
                              random_seed=RANDOM_SEED):
    """
    Loads raw images, applies transformations, splits into train/val/test sets,
    and saves them to the processed data directory.

    Args:
        raw_data_dir (str): Path to the directory containing raw data (e.g., 'data/raw/').
        processed_data_dir (str): Path to the directory where processed data will be saved.
        image_size (tuple): Target (height, width) for image resizing.
        train_ratio (float): Proportion of the dataset to include in the train split.
        val_ratio (float): Proportion of the dataset to include in the validation split.
        random_seed (int): Seed for reproducibility.
    """
    print(f"\n--- Starting Data Preprocessing and Splitting ---")
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Processed data directory: {processed_data_dir}")
    print(f"Target image size: {image_size}")
    print(f"Train/Val/Test split ratios: {train_ratio}/{val_ratio}/{1.0 - train_ratio - val_ratio}")

    # Ensure processed data directory is clean
    if os.path.exists(processed_data_dir):
        print(f"Clearing existing processed data directory: {processed_data_dir}")
        shutil.rmtree(processed_data_dir)
    os.makedirs(processed_data_dir, exist_ok=True)

    all_images = []
    all_labels = []
    class_names = []

    # Gather all image paths and their corresponding labels
    print("Collecting image paths and labels...")
    for class_name in os.listdir(raw_data_dir):
        class_path = os.path.join(raw_data_dir, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    all_images.append(os.path.join(class_path, img_name))
                    all_labels.append(class_name)

    if not all_images:
        print(f"No images found in {raw_data_dir}. Please check the path and content.")
        return

    print(f"Found {len(all_images)} images across {len(class_names)} classes.")

    # Calculate test_ratio based on remaining proportion
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError("Sum of train_ratio and val_ratio exceeds 1.0")

    # Split data into training + validation and test sets first, stratified by labels
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_images, all_labels, test_size=test_ratio, stratify=all_labels, random_state=random_seed
    )

    # Split training + validation into training and validation sets, stratified
    # Adjust val_size relative to X_train_val
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, stratify=y_train_val, random_state=random_seed
    )

    print(f"Dataset split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Create directory structure for processed data
    create_directory_structure(processed_data_dir, class_names)

    # Get transformations
    train_transform, val_test_transform = get_transforms(image_size, data_augmentation=True)

    # Function to save images to their respective split directories
    def save_images(image_paths, labels, split_name, transform):
        print(f"Processing and saving {split_name} set ({len(image_paths)} images)...")
        for i in tqdm(range(len(image_paths)), desc=f"Saving {split_name} images"):
            img_path = image_paths[i]
            label = labels[i]
            try:
                # Open image
                img = Image.open(img_path).convert('RGB')
                # Apply transform
                tensor_img = transform(img)
                # Convert back to PIL Image to save (denormalize for visual inspection if needed,
                # but saving as is for model loading is fine if using PyTorch Dataloaders)
                # For direct saving, it's often better to just resize and save without ToTensor/Normalize
                # Or save as numpy arrays if you plan to load them customly.
                # Here, we will save the raw image after resize, and let the DataLoader apply transforms later.
                # This makes the processed/ folder more readable and flexible.

                # Re-define simpler save transform for actual disk saving
                save_transform = transforms.Resize(image_size)
                img_to_save = save_transform(Image.open(img_path).convert('RGB'))

                # Define save path
                save_dir = os.path.join(processed_data_dir, split_name, label)
                # Create a unique filename to avoid overwrites (if original names repeat)
                # Using original basename for simplicity, assume unique enough within class
                base_filename = os.path.basename(img_path)
                save_path = os.path.join(save_dir, base_filename)

                img_to_save.save(save_path)
            except Exception as e:
                print(f"Error processing/saving image {img_path}: {e}")

    # Process and save images for each split
    save_images(X_train, y_train, 'train', train_transform) # Using train_transform for train set (with augmentation)
    save_images(X_val, y_val, 'val', val_test_transform)
    save_images(X_test, y_test, 'test', val_test_transform)

    print(f"--- Data Preprocessing and Splitting Complete! ---")
    print(f"Processed data available at: {processed_data_dir}")

# --- 3. Main Execution Block ---

if __name__ == "__main__":
    # Define paths relative to the project root
    RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/raw/')
    PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/processed/')

    # Make sure to run this script from the 'src' directory or adjust paths accordingly.
    # If you run it from the project root, you might need to adjust:
    # RAW_DATA_PATH = 'data/raw/'
    # PROCESSED_DATA_PATH = 'data/processed/'

    # Example usage:
    # First, ensure your raw data is in 'data/raw/' structured by class folders.
    # Then, run this script.
    preprocess_and_split_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)


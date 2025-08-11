# src/predict.py

import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image
import os
import yaml
import sys
from io import BytesIO # Import BytesIO to handle in-memory image data

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

# Load the main configuration (global scope for helper functions)
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
    Builds a pre-trained CNN model architecture.
    This function should be identical to the one in train.py and evaluate.py
    to ensure the model architecture matches the saved weights.
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=None) # No weights here; we'll load custom ones
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

# --- 4. Image Preprocessing for Prediction ---
print("\n--- Section 4: Image Preprocessing for Prediction ---")

# Modify preprocess_image to accept file-like objects (like Streamlit's UploadedFile)
def preprocess_image(image_input, image_size, mean, std):
    """
    Loads and preprocesses a single image for model prediction.
    Can accept a file path (str), bytes, or a file-like object (e.g., UploadedFile).
    """
    transform = transforms.Compose([
        transforms.Resize(tuple(image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    try:
        if isinstance(image_input, str): # If it's a file path
            image = Image.open(image_input).convert('RGB')
            print(f"Image '{os.path.basename(image_input)}' preprocessed successfully.")
        elif hasattr(image_input, 'read'): # If it's a file-like object (like UploadedFile)
            image = Image.open(image_input).convert('RGB')
            # For logging, if it has a 'name' attribute like UploadedFile does
            name_to_log = getattr(image_input, 'name', 'uploaded image')
            print(f"Image '{name_to_log}' preprocessed successfully.")
        elif isinstance(image_input, bytes): # If it's raw bytes
            image = Image.open(BytesIO(image_input)).convert('RGB')
            print(f"Bytes image preprocessed successfully.")
        else:
            raise ValueError("Unsupported image input type. Must be path, bytes, or file-like object.")

        image = transform(image).unsqueeze(0) # Add batch dimension
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# --- 5. Prediction Function ---
print("\n--- Section 5: Prediction Function ---")

def predict_image(model, image_tensor, class_names, device):
    """
    Makes a prediction on a preprocessed image tensor.
    """
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        # Get the class with the highest probability
        confidence, predicted_class_idx = torch.max(probabilities, 1)

    predicted_class_name = class_names[predicted_class_idx.item()]
    confidence_score = confidence.item() * 100

    print(f"\nPrediction Results:")
    print(f"Predicted Disease: {predicted_class_name.replace('Tomato_', '').replace('_', ' ')}")
    print(f"Confidence: {confidence_score:.2f}%")

    return predicted_class_name, confidence_score, probabilities


# --- Main Execution Block (for direct script execution) ---
if __name__ == "__main__":
    # Define paths relative to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'model.pth')

    # Ensure model file exists
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Trained model weights not found at '{MODEL_SAVE_PATH}'.")
        print("Please ensure 'src/train.py' was run successfully and saved the model.")
        sys.exit(1)

    # Reconstruct class names (order matters!)
    try:
        dummy_dataset = datasets.ImageFolder(os.path.join(project_root, 'data', 'processed', 'train'))
        class_names = dummy_dataset.classes
        print(f"Class names loaded: {class_names}")
    except Exception as e:
        print(f"Error loading class names from processed data: {e}")
        print("Please ensure your 'data/processed/train' directory exists and contains class folders.")
        sys.exit(1)

    # Build and load model
    model = build_model(train_config['model_name'], train_config['num_classes'], pretrained=False, device=device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print(f"Trained model loaded from: {MODEL_SAVE_PATH}")

    # Define common normalization values (must match training/evaluation)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # --- Example Usage ---
    # To test, you can pick an image from your test set
    example_image_path = os.path.join(project_root, 'data', 'processed', 'test', 'Tomato_healthy', 'image_0.jpg') # CHANGE THIS PATH to a real image in your test set

    if not os.path.exists(example_image_path):
        print(f"\n--- IMPORTANT ---")
        print(f"Example image path not found: {example_image_path}")
        print(f"Please replace 'example_image_path' in src/predict.py with the path to a real image from your test set (e.g., from 'data/processed/test/Tomato_Bacterial_spot/').")
        print(f"Alternatively, you can provide an image path as a command-line argument when running this script.")
        # Try to find *any* image in the test set to demonstrate
        for root, dirs, files in os.walk(os.path.join(project_root, 'data', 'processed', 'test')):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    example_image_path = os.path.join(root, file)
                    print(f"\n--- Using example image: {example_image_path} ---")
                    break
            if example_image_path != os.path.join(project_root, 'data', 'processed', 'test', 'Tomato_healthy', 'image_0.jpg'): # If we found a new path
                break
        if example_image_path == os.path.join(project_root, 'data', 'processed', 'test', 'Tomato_healthy', 'image_0.jpg') and not os.path.exists(example_image_path):
            print("\nError: No example image could be found. Cannot run prediction demonstration.")
            sys.exit(1)


    # Preprocess the image
    input_image_tensor = preprocess_image(
        example_image_path, # Now accepts path
        data_config['image_size'],
        mean,
        std
    )

    if input_image_tensor is not None:
        # Make prediction
        predicted_class, confidence, all_probabilities = predict_image(
            model,
            input_image_tensor,
            class_names,
            device
        )

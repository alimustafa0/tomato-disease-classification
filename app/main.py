# app/main.py

import streamlit as st
import os
import sys
import yaml
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets

# --- 1. Path Setup and Configuration Loading ---

# Find the project root directory dynamically
# This helps when the Streamlit app might be launched from a different directory
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, '..'))

# Add the 'src' directory to the Python path to import modules from it
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)
    # print(f"Added '{src_path}' to sys.path for model and preprocessing imports.") # For debugging

try:
    # Import necessary functions from src/predict.py
    from predict import build_model, preprocess_image, predict_image
    # print("Successfully imported build_model, preprocess_image, predict_image from src/predict.py")
except ImportError as e:
    st.error(f"Error importing prediction utilities: {e}")
    st.info("Please ensure 'src/predict.py' exists and is correctly defined.")
    st.stop() # Stop Streamlit execution if critical import fails

def load_config(config_path=os.path.join(project_root, 'params.yaml')):
    """
    Loads configuration parameters from a YAML file.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.error(f"Configuration file '{config_path}' not found.")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML file: {e}")
        st.stop()

# Load the main configuration
config = load_config()
train_config = config['train']
data_config = config['data_preprocessing']

# --- 2. Model Loading ---

@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_trained_model(model_name, num_classes, model_path, device):
    """
    Loads the trained model weights.
    """
    model = build_model(model_name, num_classes, pretrained=False, device=device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set model to evaluation mode
        return model
    except FileNotFoundError:
        st.error(f"Trained model weights not found at '{model_path}'.")
        st.info("Please ensure 'src/train.py' was run successfully and saved the model.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()

# Determine device
device = torch.device(train_config['device'] if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Using device: {device}")

# Define model save path
MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'model.pth')

# Reconstruct class names (order matters!)
# Assuming 'data/processed/train' exists and has all classes.
try:
    dummy_dataset = datasets.ImageFolder(os.path.join(project_root, 'data', 'processed', 'train'))
    class_names = dummy_dataset.classes
except Exception as e:
    st.error(f"Error loading class names from processed data: {e}")
    st.info("Please ensure your 'data/processed/train' directory exists and contains class folders.")
    st.stop()

# Load the model only once
model = load_trained_model(train_config['model_name'], train_config['num_classes'], MODEL_SAVE_PATH, device)

# Define common normalization values (must match training/evaluation/prediction)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# --- 3. Streamlit UI ---

st.set_page_config(page_title="Tomato Disease Detector", layout="centered")

st.title("🌿 Tomato Disease Detector")
st.markdown("Upload an image of a tomato leaf to detect potential diseases.")

# Sidebar for information
st.sidebar.header("About")
st.sidebar.info(
    "This application uses a pre-trained ResNet model to classify diseases "
    "in tomato plant leaves. It's built as part of a MLOps project focusing on "
    "data preprocessing, model training, evaluation, and deployment."
)
st.sidebar.header("Detected Classes")
st.sidebar.write("The model can detect the following diseases (or healthy leaves):")
for class_name in class_names:
    st.sidebar.write(f"- {class_name.replace('Tomato_', '').replace('_', ' ')}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    input_image_tensor = preprocess_image(
        uploaded_file, # Streamlit file uploader handles the file object
        data_config['image_size'],
        mean,
        std
    )

    if input_image_tensor is not None:
        # Make prediction
        predicted_class_name, confidence_score, all_probabilities = predict_image(
            model,
            input_image_tensor,
            class_names,
            device
        )

        st.success(f"Prediction: **{predicted_class_name.replace('Tomato_', '').replace('_', ' ')}**")
        st.write(f"Confidence: **{confidence_score:.2f}%**")

        st.subheader("All Class Probabilities:")
        # Display all probabilities in a nice format
        prob_dict = {name.replace('Tomato_', '').replace('_', ' '): float(prob) for name, prob in zip(class_names, all_probabilities[0])}
        st.json(prob_dict) # Or use st.bar_chart for a visual representation
        # st.bar_chart(prob_dict) # Uncomment for a bar chart of probabilities

        st.markdown("---")
        st.write("For more details, refer to the project's documentation.")


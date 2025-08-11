# Tomato Disease Classification

## Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for classifying diseases in tomato plant leaves using a Convolutional Neural Network (CNN). The system is designed to be robust and scalable, incorporating best practices for data management, model development, testing, and deployment.

---

## Project Structure

```bash
tomato-disease-classification/
tomato-disease-classification/
│
├── data/                     # Raw and processed datasets
│   ├── raw/                  # Original PlantVillage tomato images
│   ├── processed/            # Preprocessed images (train/val/test splits)
│   └── README.md             # Dataset notes
│
├── notebooks/                # Jupyter notebooks for exploration, preprocessing, training, evaluation
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb    # (To be implemented)
│   └── 04_model_evaluation.ipynb  # (To be implemented)
│
├── src/                      # Source code for ML pipeline components
│   ├── data_preprocessing.py  # Data preprocessing and augmentation logic
│   ├── train.py               # Model training script
│   ├── evaluate.py            # Model evaluation script
│   ├── predict.py             # Single image prediction script
│   └── utils.py               # Helper utilities (e.g., config loading)
│
├── models/                   # Storage for trained models and configurations
│   ├── model.pth              # Saved best model weights
│   ├── model_config.yaml      # Model architecture details and training parameters (optional, params.yaml handles much)
│   └── README.md              # Notes about model version, performance
│
├── app/                      # Web application for disease classification
│   ├── main.py                # Streamlit/FastAPI app entry point
│   ├── requirements.txt       # Python dependencies for the app
│   ├── Dockerfile             # Docker instructions to containerize the app
│   ├── config.yaml            # App-level configurations
│   └── templates/             # Frontend templates (if not using Streamlit directly)
│       └── index.html
│
├── tests/                    # Unit and integration tests
│   ├── test_data_preprocessing.py
│   ├── test_train.py              # (To be implemented)
│   ├── test_evaluate.py           # (To be implemented)
│   └── test_predict.py            # (To be implemented)
│
├── .github/                  # GitHub Actions CI/CD workflows
│   └── workflows/
│       └── ci_cd.yml          # Automated testing, linting, deployment workflow
│
├── dvc.yaml                  # Data Version Control pipeline configuration
├── params.yaml               # Stores hyperparameters and configuration values
├── .gitignore                # Files/folders to ignore in Git
├── README.md                 # Project overview, setup instructions, usage
└── LICENSE                   # Licensing terms
```

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/tomato-disease-classification.git](https://github.com/your-username/tomato-disease-classification.git)
    cd tomato-disease-classification
    ```

2.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install DVC (Data Version Control):**
    ```bash
    pip install dvc
    dvc init
    ```

4.  **Install Python Dependencies:**
    ```bash
    pip install -r app/requirements.txt
    # Also install dev dependencies for notebooks/testing
    pip install pytest pyyaml scikit-learn matplotlib seaborn tqdm
    # Install PyTorch (choose based on your CUDA/CPU setup)
    # For CUDA: pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    # For CPU: pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
    ```

---

## Data Management

1.  **Acquire Raw Data:**
    Place your original PlantVillage tomato images into the `data/raw/` directory, organized by disease class. Refer to `data/README.md` for more details on the dataset structure.

2.  **Explore Data:**
    Run the `01_data_exploration.ipynb` Jupyter notebook to visualize the dataset, check class balance, and inspect sample images.
    ```bash
    jupyter notebook notebooks/01_data_exploration.ipynb
    ```

3.  **Preprocess and Split Data:**
    Execute the `02_data_preprocessing.ipynb` notebook to preprocess (resize, augment, normalize) and split the raw images into `train`, `val`, and `test` sets, saving them to `data/processed/`.
    ```bash
    jupyter notebook notebooks/02_data_preprocessing.ipynb
    ```

---

## Model Development

1.  **Configure Training Parameters:**
    Edit `params.yaml` to adjust model hyperparameters (e.g., `batch_size`, `epochs`, `learning_rate`).

2.  **Train the Model:**
    Run the training script to train the CNN model and save the best weights to `models/model.pth`.
    ```bash
    python src/train.py
    ```

3.  **Evaluate the Model:**
    Execute the evaluation script to assess model performance on the test set, including a classification report and confusion matrix.
    ```bash
    python src/evaluate.py
    ```

4.  **Make Predictions:**
    Use the prediction script to classify a single image. Update the `example_image_path` in `src/predict.py` or provide a path as a command-line argument.
    ```bash
    python src/predict.py
    ```

---

## Application Usage

1.  **Run the Streamlit App:**
    Launch the web application to interactively upload images and get predictions.
    ```bash
    streamlit run app/main.py
    ```
    This will open the app in your web browser.

---

## Testing

Run unit tests for various components of the pipeline:
```bash
python -m unittest discover tests
```

---

## CI/CD
This project uses GitHub Actions for Continuous Integration and Continuous Deployment. The workflow defined in .github/workflows/ci_cd.yml automates:

Testing: Running unit tests on code pushes and pull requests.

Building Docker Image: Verifying that the application's Docker image can be built.

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
# app/Dockerfile

# Use a lightweight Python base image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the entire project into the container
# This copies everything from your project root into /app in the container
COPY . /app

# Install system dependencies if any (e.g., for image processing libraries)
# For PyTorch with CPU, usually no extra system dependencies are needed beyond Python.
# If using a GPU, you might need NVIDIA CUDA toolkit base images.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgl1-mesa-glx \
#     libsm6 \
#     libxrender1 \
#     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
# Using --no-cache-dir to save space in the Docker image
# Using --compile to compile .pyc files for faster startup
# Using --upgrade pip to ensure pip is up to date
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r app/requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
# We need to explicitly point to the app/main.py script
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Ensure files have executable permissions if needed (though streamlit run handles it)
# RUN chmod +x app/main.py

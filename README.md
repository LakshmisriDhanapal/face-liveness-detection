# face-liveness-detection
MINI PROJECT

# Face Liveness Detection using MobileNetV2

## Overview
This project implements a **Face Liveness Detection** model using the MobileNetV2 architecture for detecting real vs spoofed face images. The goal is to create a lightweight and efficient binary classification model capable of distinguishing between real and spoofed face images for use in edge devices like browsers or low-power environments.

## Features
- **Transfer Learning**: Utilizes MobileNetV2 pre-trained on ImageNet.
- **Data Augmentation**: Enhances model generalization with transformations.
- **Class Balancing**: Includes class weights for handling imbalanced datasets.
- **Early Stopping**: Avoids overfitting by monitoring validation loss.
- **Model Saving**: Saves the trained model in `.h5` format for reuse.

## Dataset
The project uses the **LCC FASD Dataset**, which is organized into the following structure:
```
LCC_FASD/
  LCC_FASD_training/
    real/
    spoof/
  LCC_FASD_development/
    real/
    spoof/
```
- **`real`**: Images of real faces.
- **`spoof`**: Images of spoofed faces (e.g., photos, videos, or masks).

Ensure that the dataset is placed in the correct directory structure before running the code.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/lakshmisridhanapl/face-liveness-detection.git
   cd face-liveness-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Load the Dataset
The dataset should follow the directory structure specified above. If necessary, preprocess the dataset to ensure image quality and uniformity.

### 2. Run the Training Script
To train the model:
```bash
python train_model.py
```

This script will:
- Load and preprocess the dataset.
- Train the model using MobileNetV2 as a base.
- Save the trained model as `face_liveness_detection_model.h5`.

### 3. Evaluate the Model
The script evaluates the model on the validation set and prints:
- Validation Loss
- Validation Accuracy

### 4. Visualize Training Results
The script also plots the training and validation accuracy/loss for better analysis.

## Code Structure
- `train_model.py`: Main training script.
- `requirements.txt`: Python dependencies.
- `LCC_FASD/`: Dataset folder (to be added by the user).

## Model Architecture
The model uses the following architecture:
1. **Base Model**: MobileNetV2 (pre-trained on ImageNet, frozen weights).
2. **Custom Layers**:
   - Global Average Pooling
   - Dense (128 neurons, ReLU activation)
   - Dropout (50%)
   - Output Layer (1 neuron, Sigmoid activation for binary classification)

## Key Hyperparameters
- **Image Dimensions**: 224x224 pixels.
- **Batch Size**: 32.
- **Epochs**: 5 (with early stopping).
- **Loss Function**: Binary Crossentropy.
- **Optimizer**: Adam.

## Results
- **Training Accuracy**: 92
- **Validation Accuracy**: 93

Graphs for accuracy and loss will be generated automatically after training.

## Future Work
- Extend support for additional datasets.
- Optimize for deployment on edge devices using ONNX or TensorFlow.js.
- Add active liveness detection techniques (e.g., blink detection).

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## Acknowledgments
- [LCC FASD Dataset](https://www.kaggle.com/datasets/ahmedruhshan/lcc-fasd-casia-combined)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)


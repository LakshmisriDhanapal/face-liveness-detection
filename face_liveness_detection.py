import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Constants
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32
EPOCHS = 5

# Paths
train_path = 'LCC_FASD/LCC_FASD_training'
dev_path = 'LCC_FASD/LCC_FASD_development'

def load_data(data_dir):
    images = []
    labels = []
    
    for label_type in ['real', 'spoof']:
        label_dir = os.path.join(data_dir, label_type)
        label = 0 if label_type == 'real' else 1
        
        if os.path.exists(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Unable to load image {img_path}")
                    continue
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img)
                labels.append(label)

    print(f"Loaded {len(images)} images from {data_dir}")
    return np.array(images), np.array(labels)

def create_data_generators(train_dir, dev_dir, batch_size):
    # Create ImageDataGenerator for data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        dev_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, validation_generator

def create_model(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.show()

def evaluate_model(model, validation_generator):
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

def train_model():
    # Load train and development data
    train_images, train_labels = load_data(train_path)
    dev_images, dev_labels = load_data(dev_path)

    # Normalize the images
    train_images = train_images / 255.0
    dev_images = dev_images / 255.0

    # Create data generators
    train_generator, validation_generator = create_data_generators(train_path, dev_path, BATCH_SIZE)

    # Create the model
    model = create_model()

    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Calculate steps
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        class_weight=class_weights_dict,
        callbacks=[early_stopping]
    )

    # Evaluate the model on the validation data
    evaluate_model(model, validation_generator)

    # Plot training history
    plot_training_history(history)

    # Save the model
    model.save('face_liveness_detection_model.h5')
    print("Model saved as 'face_liveness_detection_model.h5'")

if _name_ == '_main_':
    train_model()

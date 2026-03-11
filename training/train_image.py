import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# Configuration
# ==========================================
# Path to your dataset (e.g., downloaded from Kaggle)
# Expected structure:
# dataset/
# ├── real/
# │   ├── img1.jpg
# │   └── img2.jpg
# └── fake/
#     ├── img1.jpg
#     └── img2.jpg
DATASET_DIR = '../dataset/images'
MODEL_SAVE_PATH = '../models/weights/image_deepfake_model.h5'
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 10

def create_model():
    """Builds a customized Convolutional Neural Network (CNN)."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5), # Prevent overfitting
        Dense(1, activation='sigmoid') # Binary classification: Real (0) or Fake (1)
    ])
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def train():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory {DATASET_DIR} not found.")
        print("Please create it and add 'real' and 'fake' subfolders with images.")
        return

    print("Loading data...")
    # Data augmentation for better generalization
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2, # 80% training, 20% validation
        rotation_range=20,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    model = create_model()
    model.summary()

    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    # Save the model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved successfully to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()

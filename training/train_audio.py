import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D

# ==========================================
# Configuration
# ==========================================
# Expected structure:
# dataset/audio/
# ├── real/
# │   ├── audio1.wav
# └── fake/
#     ├── audio1.wav
DATASET_DIR = '../dataset/audio'
MODEL_SAVE_PATH = '../models/weights/audio_deepfake_model.h5'
EPOCHS = 15

def extract_features(file_path, max_pad_len=400):
    """
    Extracts Mel-frequency cepstral coefficients (MFCC) from an audio file.
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
            
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        return None

def load_data():
    features = []
    labels = []
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory {DATASET_DIR} not found.")
        return np.array([]), np.array([])

    categories = ['real', 'fake']
    
    for category in categories:
        class_label = 0 if category == 'real' else 1
        path = os.path.join(DATASET_DIR, category)
        
        if not os.path.exists(path):
            continue
            
        for file in os.listdir(path):
            if file.endswith('.wav') or file.endswith('.mp3'):
                file_path = os.path.join(path, file)
                data = extract_features(file_path)
                if data is not None:
                    features.append(data)
                    labels.append(class_label)
                    
    return np.array(features), np.array(labels)

def create_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # 0 = real, 1 = fake
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train():
    print("Extracting acoustic features (this may take a while)...")
    X, y = load_data()
    
    if len(X) == 0:
        print("No audio data found. Please place .wav files in the dataset folder.")
        return
        
    # Reshape for CNN
    # MFCCs are 40 x 400. We'll treat 400 as timesteps and 40 as features for Conv1D, or vice versa
    X = np.swapaxes(X, 1, 2) # Now shape is (samples, 400, 40)
    
    # Simple train test split (80/20)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]
    
    model = create_model((X.shape[1], X.shape[2]))
    model.summary()
    
    print("Training audio model...")
    model.fit(X_train, y_train, batch_size=16, epochs=EPOCHS, validation_data=(X_test, y_test))
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Audio model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()

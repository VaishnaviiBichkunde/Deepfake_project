# Training the Deepfake Models

To make this a complete, functional Machine Learning project, you need to train the models on a dataset. The Flask web application is already set up to use the `.h5` model files once they are generated.

## 1. Get the Datasets from Kaggle
Download a deepfake dataset. A popular one is the [Deepfake Detection Challenge (DFDC)](https://www.kaggle.com/c/deepfake-detection-challenge) or the [140k Real and Fake Faces Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces).

Organize your downloaded data inside a `dataset/` folder at the root of your project:

```text
deepfake_project/
├── dataset/
│   ├── images/
│   │   ├── real/  (Put real face images here)
│   │   └── fake/  (Put AI-generated face images here)
│   └── audio/
│       ├── real/  (Put authentic human voice .wav files here)
│       └── fake/  (Put AI-generated voice .wav files here)
```

## 2. Train the Image CNN Model
Once your `dataset/images` folder is populated:

1. Open your terminal.
2. Navigate to the `training` folder: `cd training`
3. Run: `python train_image.py`

This will train the Convolutional Neural Network (CNN) and save the weights to `models/weights/image_deepfake_model.h5`.

## 3. Train the Audio Model
Once your `dataset/audio` folder is populated:

1. Navigate to the `training` folder: `cd training`
2. Run: `python train_audio.py`

This extracts Mel-frequency cepstral coefficients (MFCCs) using librosa and trains a Conv1D network. It will save the file to `models/weights/audio_deepfake_model.h5`.

## 4. Run the Full App
The Flask app (`app.py`) is already configured to look for these trained `.h5` files in the `models/weights/` directory. If it finds them, it will use real Machine Learning predictions! If not, it falls back to the simulated stubs for demonstration purposes.

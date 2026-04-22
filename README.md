# Abnormal Event Detection System

A deep learning web application for detecting abnormal events in surveillance videos using **MobileNetV2** for feature extraction and **LSTM** for temporal sequence analysis.

## Architecture

```
Video Frames → MobileNetV2 (1280-dim features) → LSTM → Sigmoid → Abnormal/Normal
```

## Features

- **Real-time Video Analysis**: Upload any video and get instant anomaly detection
- **Frame-Level Predictions**: Per-frame probability scores
- **Visual Timeline**: Probability timeline with threshold visualization
- **Multiple Backbone Comparison**: MobileNetV2, VGG16, ResNet50 tested
- **Lightweight Model**: ~20MB model size for fast inference

## Why MobileNetV2?

| Model | Size | Parameters | Inference Speed |
|-------|------|------------|----------------|
| **MobileNetV2 + LSTM** | ~20 MB | ~3.5M | Fast |
| ResNet50 + LSTM | ~29 MB | ~25M | Medium |
| VGG16 + LSTM | ~11 MB | ~138M | Slow |

MobileNetV2 provides the best balance of accuracy, speed, and model size.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Deployment

This app can be deployed on [Streamlit Cloud](https://streamlit.io/cloud).

## Project Structure

```
AJJJ/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/                 # Trained models
│   ├── mobilenet_lstm_abnormal.keras
│   ├── mobilenet_lstm_weights.weights.h5
│   ├── resnet50_lstm_abnormal.keras
│   └── vgg16_lstm_abnormal.keras
├── notebooks/              # Training and experimentation
│   ├── Avenue_Abnormal_Detection_MobileNet_LSTM (1).ipynb
│   ├── avenue_resnet50_lstm.ipynb
│   ├── avenue_vgg16_lstm.ipynb
│   ├── avenue_xai_notebook.ipynb
│   └── compare_backbones.ipynb
├── src/                    # Utility scripts
│   ├── retrain_model.py    # Model retraining
│   └── save_weights.py     # Weight conversion
└── assets/                 # Data and results
    ├── backbone_comparison.csv
    └── all_predictions_summary.csv
```

## Model Details

### MobileNetV2 Feature Extractor
- Input: 224x224x3 RGB frames
- Output: 1280-dimensional feature vectors
- Pre-trained on ImageNet, frozen during training

### LSTM Temporal Analyzer
- Input: Sequence of 16 frames (16x1280)
- Architecture: LSTM(256) → Dropout → LSTM(128) → Dense(64) → Sigmoid
- Output: Anomaly probability per frame

## Dataset

Trained and tested on the **Avenue Dataset** for abnormal event detection in surveillance videos.

## License

MIT License

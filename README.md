# Abnormal Event Detector

A deep learning web application for detecting abnormal events in surveillance videos using **MobileNetV2** for feature extraction and **LSTM** for temporal sequence analysis.

## Architecture

```
Video Frames → MobileNetV2 (1280-dim features) → LSTM → Sigmoid → Abnormal/Normal
```

## Features

- **Real-time Video Analysis**: Upload any video and get instant anomaly detection
- **Frame-Level Predictions**: Per-frame probability scores
- **Visual Timeline**: Probability timeline with threshold visualization
- **Multiple Backbone Comparison**: MobileNetV2 (used), VGG16, ResNet50 tested
- **Lightweight Model**: ~20MB model size for fast inference

## Why MobileNetV2?

| Model | Size | Parameters | Inference Speed |
|-------|------|------------|-----------------|
| **MobileNetV2 + LSTM** | ~20 MB | ~3.5M | ⚡ Fast |
| ResNet50 + LSTM | ~29 MB | ~25M | Medium |
| VGG16 + LSTM | ~11 MB | ~138M | Slow |

MobileNetV2 provides the best balance of accuracy, speed, and model size.

## Demo

Deployed on [Streamlit Cloud](https://share.streamlit.io)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Deployment

This app is deployed on [Streamlit Cloud](https://share.streamlit.io).

## Dataset

Trained and tested on the **Avenue Dataset** for abnormal event detection.

## License

MIT License

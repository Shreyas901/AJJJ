import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

SEQ_LENGTH = 16

inputs = Input(shape=(SEQ_LENGTH, 1280))
x = LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(inputs)
x = Dropout(0.4)(x)
x = LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

BASE_DIR = r"C:\Users\awati\Avenue_Dataset 3\Avenue"
MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_lstm_abnormal.keras")
WEIGHTS_PATH = os.path.join(BASE_DIR, "mobilenet_lstm_weights.weights.h5")

model.load_weights(MODEL_PATH)
model.save_weights(WEIGHTS_PATH)
print(f"Weights saved to: {WEIGHTS_PATH}")

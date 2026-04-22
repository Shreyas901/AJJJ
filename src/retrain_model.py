import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = r"C:\Users\awati\Avenue_Dataset 3\Avenue"
MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_lstm_abnormal.keras")
CACHE_DIR = os.path.join(BASE_DIR, "feature_cache")

SEQ_LENGTH = 16
STEP_SIZE = 8
VAL_SPLIT = 0.15
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4

print("Loading cached features...")
train_videos = [f"{i:02d}_features.npy" for i in range(1, 17)]

all_features = []
all_labels = []

for vname in train_videos:
    cache_path = os.path.join(CACHE_DIR, vname)
    if os.path.exists(cache_path):
        feats = np.load(cache_path)
        seqs = []
        for start in range(0, len(feats) - SEQ_LENGTH + 1, STEP_SIZE):
            seqs.append(feats[start:start+SEQ_LENGTH])
        if len(seqs) > 0:
            all_features.extend(seqs)
            all_labels.extend([0] * len(seqs))

X = np.array(all_features)
y = np.array(all_labels)
print(f"Loaded {len(X)} normal sequences")

def create_synthetic_abnormal(normal_seqs, noise_factor=2.5):
    n_samples = len(normal_seqs)
    abnormal = normal_seqs.copy()
    noise = np.random.normal(0, noise_factor * abnormal.std(), abnormal.shape)
    abnormal += noise
    for i in range(len(abnormal)):
        perm = np.random.permutation(SEQ_LENGTH)
        abnormal[i] = abnormal[i][perm]
    return abnormal

X_normal = X
X_abnormal = create_synthetic_abnormal(X_normal)
y_abnormal = np.ones(len(X_abnormal), dtype=np.float32)
y_normal = np.zeros(len(X_normal), dtype=np.float32)

X_all = np.vstack([X_normal, X_abnormal])
y_all = np.hstack([y_normal, y_abnormal])

shuffle_idx = np.random.permutation(len(X_all))
X_all = X_all[shuffle_idx]
y_all = y_all[shuffle_idx]

X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=VAL_SPLIT, random_state=42, stratify=y_all)
print(f"Training: {X_tr.shape}, Validation: {X_val.shape}")

def build_lstm_model():
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
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

print("Building model...")
model = build_lstm_model()
model.summary()

print("Training...")
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'),
        ModelCheckpoint(MODEL_PATH, monitor='val_auc', mode='max', save_best_only=True)
    ],
    verbose=1
)

print(f"Model saved to: {MODEL_PATH}")
print("Done!")

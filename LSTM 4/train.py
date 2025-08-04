import os
import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from preprocess_data import preprocess_data

# ————— LOAD DATA —————
x_train, x_test, y_train, y_test = preprocess_data()

# ————— HYPERPARAMETERS —————
SEQ_LEN    = x_train.shape[1]   # should be 90
FEATURES   = x_train.shape[2]   # 99
N_CLASSES  = y_train.shape[1]   # number of detected actions

# ————— BUILD MODEL —————
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, FEATURES)),
    BatchNormalization(),

    LSTM(128, return_sequences=True),
    Dropout(0.3),

    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dense(N_CLASSES, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

# ————— CALLBACKS —————
log_dir = os.path.join('Logs')
callbacks = [
    TensorBoard(log_dir=log_dir),
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_action_lstm.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# ————— TRAIN —————
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=300,
    callbacks=callbacks
)

# ————— SUMMARY & SAVE —————
model.summary()
model.save("action_lstm_final.h5")

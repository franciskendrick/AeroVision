from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout
import numpy as np

# 1) Load preprocessed data
x_train = np.load("D:\Programming Projects\Repositories\AeroVision\Angles/npy files/x_train.npy")
x_test  = np.load("D:\Programming Projects\Repositories\AeroVision\Angles/npy files/x_test.npy")
y_train = np.load("D:\Programming Projects\Repositories\AeroVision\Angles/npy files/y_train.npy")
y_test  = np.load("D:\Programming Projects\Repositories\AeroVision\Angles/npy files/y_test.npy")

sequence_length, num_features = x_train.shape[1], x_train.shape[2]

# 2) Build the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu',
               input_shape=(sequence_length, num_features)))
model.add(Dropout(0.3))

model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Final output layer: 3 classes
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 3) Train
history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=8,
                    validation_data=(x_test, y_test))

# 4) Save the trained model
model.save("action.h5")

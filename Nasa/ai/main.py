# ==========================================
# CNN + LSTM Model for NASA Satellite Images
# ==========================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split

# -----------------------------
# 1️⃣ Load & Preprocess Images
# -----------------------------

# Assume you have a folder with subfolders (e.g., 'cloudy', 'clear', 'storm')
# Example path: ./nasa_dataset/
data_dir = './nasa_dataset/'  # put your NASA image path here
img_size = (128, 128)  # Resize all images to this size
sequence_length = 5     # number of frames per sequence (for LSTM)

def load_image_sequences(base_path, img_size=(128, 128), sequence_length=5):
    X, y = [], []
    class_labels = os.listdir(base_path)
    
    for label in class_labels:
        folder = os.path.join(base_path, label)
        images = sorted(os.listdir(folder))
        
        # Convert each image to array
        image_arrays = [img_to_array(load_img(os.path.join(folder, img), target_size=img_size)) / 255.0
                        for img in images]
        
        # Create sliding window sequences for LSTM
        for i in range(len(image_arrays) - sequence_length + 1):
            seq = image_arrays[i:i+sequence_length]
            X.append(seq)
            y.append(class_labels.index(label))
    
    return np.array(X), np.array(y)

print("🔄 Loading NASA image sequences...")
X, y = load_image_sequences(data_dir, img_size, sequence_length)
print(f"✅ Loaded {len(X)} sequences of shape {X.shape}")

# -----------------------------
# 2️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 3️⃣ Build CNN + LSTM Model
# -----------------------------
input_shape = (sequence_length, img_size[0], img_size[1], 3)

cnn_base = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten()
])

# Combine CNN and LSTM
model_input = Input(shape=input_shape)
cnn_features = TimeDistributed(cnn_base)(model_input)
lstm_out = LSTM(128, return_sequences=False)(cnn_features)
drop = Dropout(0.3)(lstm_out)
output = Dense(len(np.unique(y)), activation='softmax')(drop)

model = Model(inputs=model_input, outputs=output)

# -----------------------------
# 4️⃣ Compile Model
# -----------------------------
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# 5️⃣ Train Model
# -----------------------------
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=8)

# -----------------------------
# 6️⃣ Evaluate Model
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.3f}")

# -----------------------------
# 7️⃣ Make Prediction
# -----------------------------
sample = X_test[0:1]  # take one sample sequence
pred = model.predict(sample)
predicted_class = np.argmax(pred)
print(f"🛰 Predicted class: {predicted_class}, Actual: {y_test[0]}")

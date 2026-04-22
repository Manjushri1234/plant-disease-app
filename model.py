# model.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------

# Dataset Path

# -----------------------------

train_dir = "PlantVillage"

img_size = 128
batch_size = 32

# -----------------------------

# Data Augmentation

# -----------------------------

datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=20,
zoom_range=0.2,
shear_range=0.2,
horizontal_flip=True,
validation_split=0.2
)

train_generator = datagen.flow_from_directory(
train_dir,
target_size=(img_size, img_size),
batch_size=batch_size,
class_mode="categorical",
subset="training"
)

val_generator = datagen.flow_from_directory(
train_dir,
target_size=(img_size, img_size),
batch_size=batch_size,
class_mode="categorical",
subset="validation"
)

# -----------------------------

# CNN Model

# -----------------------------

model = Sequential([


Conv2D(32, (3,3), activation="relu", input_shape=(img_size, img_size, 3)),
BatchNormalization(),
MaxPooling2D(2,2),

Conv2D(64, (3,3), activation="relu"),
BatchNormalization(),
MaxPooling2D(2,2),

Conv2D(128, (3,3), activation="relu"),
BatchNormalization(),
MaxPooling2D(2,2),

Flatten(),
Dropout(0.5),

Dense(256, activation="relu"),
Dropout(0.3),

Dense(train_generator.num_classes, activation="softmax")


])

# -----------------------------

# Compile Model

# -----------------------------

model.compile(
optimizer="adam",
loss="categorical_crossentropy",
metrics=["accuracy"]
)

model.summary()

# -----------------------------

# Callbacks

# -----------------------------

early_stop = EarlyStopping(
monitor="val_loss",
patience=3,
restore_best_weights=True
)

checkpoint = ModelCheckpoint(
"plant_disease_model.h5",
monitor="val_accuracy",
save_best_only=True
)

# -----------------------------

# Train Model

# -----------------------------

history = model.fit(
train_generator,
validation_data=val_generator,
epochs=15,
callbacks=[early_stop, checkpoint]
)

print("Training Completed")

# -----------------------------

# Save Final Model

# -----------------------------

model.save("plant_disease_model.h5")
print("Model saved as plant_disease_model.h5")

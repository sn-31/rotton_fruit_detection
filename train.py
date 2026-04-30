"""
FruitSense — Training Script
Trains a CNN on the Kaggle fresh/rotten fruit dataset and saves the model.
Run: python train.py
"""

import os
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = (150, 150)
BATCH_SIZE  = 32
EPOCHS      = 10
MODEL_PATH  = "fruit_model.keras"
META_PATH   = "model_meta.json"

# ── Dataset ───────────────────────────────────────────────────────────────────
print("⬇  Downloading dataset...")
path = kagglehub.dataset_download("sriramr/fruits-fresh-and-rotten-for-classification")
train_dir = os.path.join(path, "dataset", "train")
test_dir  = os.path.join(path, "dataset", "test")

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)

train = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    shuffle=True,
)

val = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
)

# Canonical order that the frontend expects
CLASS_NAMES = ["freshapples", "freshbanana", "freshoranges",
               "rottenapples", "rottenbanana", "rottenoranges"]

# Verify dataset classes match what we expect
detected = list(train.class_indices.keys())
print(f"✅ Dataset classes : {detected}")
assert set(detected) == set(CLASS_NAMES), \
    f"Class mismatch! Expected {CLASS_NAMES}, got {detected}"

NUM_CLASSES = len(CLASS_NAMES)

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                  input_shape=(*IMG_SIZE, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 3
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Head
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation="softmax"),
], name="FruitSense_CNN")

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1
    ),
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n🚀 Training...")
history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(val, verbose=0)
print(f"\n📊 Val Loss: {loss:.4f}  |  Val Accuracy: {acc*100:.2f}%")

# ── Save metadata (class order for the API server) ────────────────────────────
# We store the class index mapping so the server never guesses the order.
meta = {
    "class_names": CLASS_NAMES,
    "class_indices": train.class_indices,   # {"freshapples": 0, ...}
    "img_size": list(IMG_SIZE),
    "val_accuracy": round(acc, 4),
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"✅ Model saved  → {MODEL_PATH}")
print(f"✅ Metadata saved → {META_PATH}")

import kagglehub
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Download Dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("sriramr/fruits-fresh-and-rotten-for-classification")
train_dir = os.path.join(path, 'dataset', 'train')

# 2. Data Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    train_dir, 
    target_size=(150,150), 
    batch_size=32, 
    subset='training'
)

val = datagen.flow_from_directory(
    train_dir, 
    target_size=(150,150), 
    batch_size=32, 
    subset='validation'
)

class_names = list(train.class_indices.keys())
print("Classes identified:", class_names)

# 3. Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(train.num_classes,activation='softmax')
])

# 4. Training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting training...")
model.fit(train, validation_data=val, epochs=5)

# 5. Evaluation & Saving
loss, acc = model.evaluate(val)
print(f"Validation Accuracy: {acc*100:.2f}%")

model.save("fruit_model.keras")
print("Model saved as fruit_model.keras")

import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

def image_generator(folder):
    valid_extensions = ('.jpg')
    for filename in os.listdir(folder):
        if filename.lower().endswith(valid_extensions):
            try:
                img_path = os.path.join(folder, filename)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0  # Normalize here
                yield img_array
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

def create_dataset(folder, label, batch_size=32):
    dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(folder),
        output_signature=tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)
    )
    dataset = dataset.map(lambda x: (x, label))
    dataset = dataset.batch(batch_size)
    return dataset

# Set up paths
train_base_path = "C:\\Users\\M V S Akhil Teja\\VeriFace\\Test"
train_real_path = os.path.join(train_base_path, "Real")
train_fake_path = os.path.join(train_base_path, "Fake")

# Verify directories
all_paths = [train_real_path, train_fake_path]
for path in all_paths:
    if not os.path.exists(path):
        raise ValueError(f"Directory not found: {path}")

# Create and combine datasets
print("Creating training datasets...")
train_real_dataset = create_dataset(train_real_path, 0)
train_fake_dataset = create_dataset(train_fake_path, 1)
train_dataset = train_real_dataset.concatenate(train_fake_dataset)
train_dataset = train_dataset.shuffle(buffer_size=1000)

# Split dataset
dataset_size = sum(1 for _ in train_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset = train_dataset.take(train_size)
remaining_dataset = train_dataset.skip(train_size)
val_dataset = remaining_dataset.take(val_size)
test_dataset = remaining_dataset.skip(val_size)

# Preprocess data
def preprocess_data(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.clip_by_value(image, 0, 1.0)  # Clip values between 0 and 1
    return image, label

train_dataset = train_dataset.map(preprocess_data)
val_dataset = val_dataset.map(preprocess_data)
test_dataset = test_dataset.map(preprocess_data)

# Performance optimization
train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)

# Build model
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# Train model with error handling
try:
    history = model.fit(
        train_dataset,
        epochs=20,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
except Exception as e:
    print(f"Error training model: {str(e)}")
    raise

# Safe evaluation function
def safe_evaluate(model, dataset):
    try:
        test_loss, test_acc = model.evaluate(dataset)
        return test_loss, test_acc
    except ValueError as e:
        print(f"Evaluation error: {e}")
        return None, None

# Evaluate and print results
test_loss, test_acc = safe_evaluate(model, test_dataset)
if test_acc is not None:
    print(f'Final Test Accuracy: {test_acc*100:.2f}%')

# Save model
model.save('fake_face_detection_model.keras')

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Print final metrics
print("\nFinal Results:")
print(f"Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
if test_acc is not None:
    print(f"Test Accuracy: {test_acc*100:.2f}%")

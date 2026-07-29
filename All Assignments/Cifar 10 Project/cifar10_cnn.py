# ============================================================
# CIFAR-10 Image Classification using CNN (Convolutional Neural Network)
# ============================================================
# CIFAR-10 has 60,000 images across 10 categories:
# Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
#
# INSTALL REQUIRED LIBRARIES (run in terminal):
#   pip install tensorflow numpy matplotlib scikit-learn
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# TensorFlow & Keras — the deep learning framework
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────
# STEP 1: Load & Explore the Dataset
# ─────────────────────────────────────────────

# Class names for the 10 CIFAR-10 categories
CLASS_NAMES = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# Load CIFAR-10 directly from Keras (downloads automatically ~170MB)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Dataset loaded!")
print(f"  Training images : {X_train.shape}")   # (50000, 32, 32, 3)
print(f"  Test images     : {X_test.shape}")     # (10000, 32, 32, 3)
print(f"  Image size      : 32x32 pixels, 3 color channels (RGB)")
print(f"  Classes         : {len(CLASS_NAMES)}")

# ─────────────────────────────────────────────
# STEP 2: Preprocess the Data
# ─────────────────────────────────────────────

# Normalize pixel values from [0–255] to [0–1]
# This helps the neural network learn faster and more stably
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

# Convert labels to one-hot encoding
# e.g., class 3 (Cat) → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train_ohe = to_categorical(y_train, num_classes=10)
y_test_ohe  = to_categorical(y_test,  num_classes=10)

print("\nPreprocessing done!")
print(f"  Pixel range after normalization: [{X_train.min():.1f}, {X_train.max():.1f}]")

# ─────────────────────────────────────────────
# STEP 3: Visualize Sample Images
# ─────────────────────────────────────────────

def show_sample_images(X, y, num=10):
    """Display a grid of sample images from the dataset."""
    plt.figure(figsize=(15, 2))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(X[i])
        plt.title(CLASS_NAMES[y[i][0]], fontsize=8)
        plt.axis('off')
    plt.suptitle("Sample Images from CIFAR-10", fontsize=12)
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=100)
    plt.show()
    print("  Saved: sample_images.png")

show_sample_images(X_train, y_train)

# ─────────────────────────────────────────────
# STEP 4: Build the CNN Model
# ─────────────────────────────────────────────
#
# A CNN (Convolutional Neural Network) has three main building blocks:
#
#   Conv2D     — learns to detect features (edges, shapes, patterns)
#   MaxPooling — reduces image size while keeping important info
#   Dense      — fully connected layer for final decision-making
#
# Our architecture:
#   Input (32×32×3)
#    → [Conv → Conv → MaxPool → Dropout] × 2
#    → Flatten
#    → Dense(256) → Dropout → Dense(10, softmax)

def build_cnn_model():
    model = models.Sequential(name="CIFAR10_CNN")

    # ── Block 1: First pair of Conv layers ──
    model.add(layers.Input(shape=(32, 32, 3)))

    model.add(layers.Conv2D(
        filters=32,          # Learn 32 different feature detectors
        kernel_size=(3, 3),  # Each detector looks at a 3×3 patch
        padding='same',      # Keep output the same size as input
        activation='relu'    # ReLU: keeps positive values, zeroes negatives
    ))
    model.add(layers.BatchNormalization())  # Stabilizes training

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 32×32 → 16×16
    model.add(layers.Dropout(0.25))  # Randomly drop 25% of neurons to prevent overfitting

    # ── Block 2: More powerful Conv layers ──
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 16×16 → 8×8
    model.add(layers.Dropout(0.25))

    # ── Block 3: Even deeper features ──
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 8×8 → 4×4
    model.add(layers.Dropout(0.25))

    # ── Classification Head ──
    model.add(layers.Flatten())          # Convert 4×4×128 grid → 1D vector of 2048 numbers

    model.add(layers.Dense(256, activation='relu'))  # Learn complex combinations
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))       # 50% dropout before final layer

    model.add(layers.Dense(10, activation='softmax'))
    # softmax: outputs 10 probabilities that sum to 1.0
    # e.g., [0.01, 0.02, 0.85, 0.01, ...] → "85% sure it's class 2 (Bird)"

    return model

model = build_cnn_model()
model.summary()  # Print the architecture

# ─────────────────────────────────────────────
# STEP 5: Compile the Model
# ─────────────────────────────────────────────

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    # Adam: smart optimizer that adjusts learning rate automatically

    loss='categorical_crossentropy',
    # Loss function for multi-class problems (measures prediction error)

    metrics=['accuracy']
    # What we track during training
)

# ─────────────────────────────────────────────
# STEP 6: Data Augmentation
# ─────────────────────────────────────────────
# Artificially create more training data by randomly flipping/rotating images
# This helps the model generalize better (not just memorize training data)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),          # Flip image left-right (50% chance)
    layers.RandomRotation(0.1),               # Rotate up to ±10%
    layers.RandomZoom(0.1),                   # Zoom in/out up to 10%
    layers.RandomTranslation(0.1, 0.1),       # Shift image slightly
])

# ─────────────────────────────────────────────
# STEP 7: Train the Model
# ─────────────────────────────────────────────

EPOCHS = 50       # Number of times to go through the full training set
BATCH_SIZE = 64   # Number of images to process at once

# Callbacks: actions to take during training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,          # Stop if no improvement for 10 epochs
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,           # Halve learning rate when stuck
        patience=5,
        min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_cifar10_model.keras',
        monitor='val_accuracy',
        save_best_only=True   # Only save when accuracy improves
    )
]

print("\nStarting training... (this may take 10–30 minutes depending on your hardware)")
print("Tip: GPU will be much faster. Check with: print(tf.config.list_physical_devices('GPU'))\n")

# Build augmented dataset
train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train_ohe))
    .shuffle(50000)
    .batch(BATCH_SIZE)
    .map(lambda x, y: (data_augmentation(x, training=True), y),
         num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test_ohe))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks
)

# ─────────────────────────────────────────────
# STEP 8: Evaluate & Plot Results
# ─────────────────────────────────────────────

def plot_training_history(history):
    """Plot accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'],     label='Train Accuracy', color='steelblue')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy',   color='darkorange', linestyle='--')
    ax1.set_title('Model Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history.history['loss'],     label='Train Loss', color='steelblue')
    ax2.plot(history.history['val_loss'], label='Val Loss',   color='darkorange', linestyle='--')
    ax2.set_title('Model Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100)
    plt.show()
    print("  Saved: training_history.png")

plot_training_history(history)

# Final test accuracy
test_loss, test_acc = model.evaluate(val_dataset, verbose=0)
print(f"\n{'='*40}")
print(f"Final Test Accuracy : {test_acc * 100:.2f}%")
print(f"Final Test Loss     : {test_loss:.4f}")
print(f"{'='*40}")

# ─────────────────────────────────────────────
# STEP 9: Confusion Matrix & Classification Report
# ─────────────────────────────────────────────

def evaluate_detailed(model, X_test, y_test):
    """Show per-class accuracy and confusion matrix."""
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test.flatten()

    print("\nPer-class Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.title('Confusion Matrix\n(rows = true label, cols = predicted label)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.show()
    print("  Saved: confusion_matrix.png")

evaluate_detailed(model, X_test, y_test)

# ─────────────────────────────────────────────
# STEP 10: Predict on Individual Images
# ─────────────────────────────────────────────

def predict_single_image(model, X_test, y_test, index=0):
    """Show prediction for a single test image."""
    img = X_test[index]
    true_label = CLASS_NAMES[y_test[index][0]]

    # Expand dims: model expects batch → (1, 32, 32, 3)
    pred_probs = model.predict(img[np.newaxis, ...], verbose=0)[0]
    pred_label = CLASS_NAMES[np.argmax(pred_probs)]
    confidence = np.max(pred_probs) * 100

    # Display
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    color = 'green' if pred_label == true_label else 'red'
    plt.title(f"True: {true_label}\nPredicted: {pred_label} ({confidence:.1f}%)", color=color)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    colors = ['green' if i == np.argmax(pred_probs) else 'steelblue' for i in range(10)]
    plt.barh(CLASS_NAMES, pred_probs * 100, color=colors)
    plt.xlabel('Confidence (%)')
    plt.title('Prediction Probabilities')
    plt.xlim(0, 100)

    plt.tight_layout()
    plt.savefig(f'prediction_{index}.png', dpi=100)
    plt.show()

# Try predictions on a few test images
for idx in [0, 5, 10, 20, 42]:
    predict_single_image(model, X_test, y_test, index=idx)

# ─────────────────────────────────────────────
# STEP 11: Save the Final Model
# ─────────────────────────────────────────────

model.save('cifar10_final_model.keras')
print("\nModel saved as: cifar10_final_model.keras")
print("Load it later with: model = tf.keras.models.load_model('cifar10_final_model.keras')")

# ─────────────────────────────────────────────
# QUICK SUMMARY — What we built
# ─────────────────────────────────────────────
# Architecture : 3-block CNN + fully connected head
# Total Params : ~500K parameters
# Training Data: 50,000 images (augmented)
# Test Data    : 10,000 images
# Expected Acc : ~82–88% on test set
# ─────────────────────────────────────────────

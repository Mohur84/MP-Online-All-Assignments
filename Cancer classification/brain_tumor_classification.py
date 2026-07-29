# ============================================================
#  BRAIN TUMOR MRI CLASSIFICATION USING CNN
#  Framework : TensorFlow / Keras
#  Classes   : Glioma | Meningioma | Pituitary | No Tumor
#  Dataset   : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
# ============================================================

# ─────────────────────────────────────────────
# STEP 1 : IMPORT LIBRARIES
# ─────────────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow version:", tf.__version__)

# Reproducibility
tf.keras.utils.set_random_seed(42)


# ─────────────────────────────────────────────
# STEP 2 : CONFIGURATION (change paths here)
# ─────────────────────────────────────────────
TRAIN_DIR   = "dataset/Training"
TEST_DIR    = "dataset/Testing"

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 40
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
BEST_MODEL_PATH = "best_brain_tumor_model.keras"


# ─────────────────────────────────────────────
# STEP 3 : DATA LOADING & AUGMENTATION
# ─────────────────────────────────────────────
# EfficientNet expects preprocess_input (NOT simple /255). Using /255 with
# ImageNet weights prevents the backbone from producing useful features.

train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range           = 20,
    zoom_range               = 0.15,
    horizontal_flip          = True,
    width_shift_range        = 0.1,
    height_shift_range       = 0.1,
    shear_range              = 0.05,
    brightness_range         = [0.9, 1.1],
    validation_split         = 0.15
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size  = (IMG_SIZE, IMG_SIZE),
    batch_size   = BATCH_SIZE,
    class_mode   = "categorical",
    subset       = "training",
    shuffle      = True,
    seed         = 42,
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size  = (IMG_SIZE, IMG_SIZE),
    batch_size   = BATCH_SIZE,
    class_mode   = "categorical",
    subset       = "validation",
    shuffle      = False,
    seed         = 42,
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size  = (IMG_SIZE, IMG_SIZE),
    batch_size   = BATCH_SIZE,
    class_mode   = "categorical",
    shuffle      = False,
)

print("\nClass indices:", train_generator.class_indices)


# ─────────────────────────────────────────────
# STEP 4 : VISUALIZE SAMPLE IMAGES
# ─────────────────────────────────────────────
def show_sample_images(generator, class_names, n=8):
    images, labels = next(generator)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Sample Training MRI Images", fontsize=16, fontweight="bold")
    for i, ax in enumerate(axes.flatten()):
        if i < n:
            display = np.clip((images[i] + 1.0) / 2.0, 0.0, 1.0)
            ax.imshow(display)
            class_idx = np.argmax(labels[i])
            ax.set_title(class_names[class_idx], fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=120)
    plt.close()
    print("Saved: sample_images.png")

show_sample_images(train_generator, CLASS_NAMES)


# ─────────────────────────────────────────────
# STEP 5 : BUILD THE CNN MODEL
# ─────────────────────────────────────────────
def build_model(num_classes, img_size):
    base_model = EfficientNetB0(
        weights     = "imagenet",
        include_top = False,
        input_shape = (img_size, img_size, 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base_model


def set_fine_tune_layers(base_model, trainable_from):
    base_model.trainable = True
    for layer in base_model.layers[:trainable_from]:
        layer.trainable = False
    # Keep BatchNorm frozen during fine-tuning for stability.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


model, base_model = build_model(NUM_CLASSES, IMG_SIZE)

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-3),
    loss      = keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics   = ["accuracy"],
)

model.summary()


# ─────────────────────────────────────────────
# STEP 6 : CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
        verbose=1,
        mode="max",
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.4,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        mode="max",
    ),
]


# ─────────────────────────────────────────────
# STEP 7 : PHASE 1 TRAINING (frozen base)
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("PHASE 1: Training classification head only")
print("=" * 50)

history1 = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs          = 12,
    callbacks       = callbacks,
)


# ─────────────────────────────────────────────
# STEP 8 : PHASE 2 FINE-TUNING
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("PHASE 2: Fine-tuning EfficientNet layers")
print("=" * 50)

set_fine_tune_layers(base_model, trainable_from=len(base_model.layers) - 60)

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-5),
    loss      = keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics   = ["accuracy"],
)

history2 = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs          = EPOCHS,
    callbacks       = callbacks,
)


# ─────────────────────────────────────────────
# STEP 9 : PLOT TRAINING CURVES
# ─────────────────────────────────────────────
def plot_history(h1, h2):
    acc      = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss     = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]
    phase_split = len(h1.history["accuracy"])
    epochs_range = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Brain Tumor CNN — Training History", fontsize=15, fontweight="bold")

    for ax, metric, val_metric, title in zip(
        axes,
        [acc, loss],
        [val_acc, val_loss],
        ["Accuracy", "Loss"],
    ):
        ax.plot(epochs_range, metric, label="Train")
        ax.plot(epochs_range, val_metric, label="Validation")
        ax.axvline(x=phase_split, color="gray", linestyle="--", label="Fine-tune start")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    plt.close()
    print("Saved: training_curves.png")


plot_history(history1, history2)


# ─────────────────────────────────────────────
# STEP 10 : EVALUATE ON TEST SET (best checkpoint)
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("FINAL EVALUATION ON TEST SET")
print("=" * 50)

best_model = keras.models.load_model(BEST_MODEL_PATH)

test_loss, test_acc = best_model.evaluate(test_generator)
print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

test_generator.reset()
y_pred_prob = best_model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_generator.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# ─────────────────────────────────────────────
# STEP 11 : CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix — Brain Tumor Classification", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".1f"],
        ["Raw Counts", "Percentage (%)"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.close()
    print("Saved: confusion_matrix.png")


plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)


# ─────────────────────────────────────────────
# STEP 12 : PREDICT A SINGLE IMAGE
# ─────────────────────────────────────────────
def predict_single_image(image_path, model, class_names, img_size=224):
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
    img_array = preprocess_input(keras.preprocessing.image.img_to_array(img))
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    plt.figure(figsize=(5, 5))
    plt.imshow(keras.preprocessing.image.load_img(image_path))
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.1f}%", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("\nPrediction Breakdown:")
    for name, prob in zip(class_names, predictions):
        bar = "█" * int(prob * 30)
        print(f"  {name:<12}: {bar} {prob * 100:.1f}%")

    return predicted_class, confidence


# ─────────────────────────────────────────────
# STEP 13 : SAVE THE FINAL MODEL
# ─────────────────────────────────────────────
best_model.save("brain_tumor_classifier_final.keras")
print("\nModel saved as: brain_tumor_classifier_final.keras")
print(f"Best checkpoint: {BEST_MODEL_PATH}")

print("""
═══════════════════════════════════════════════
  PROJECT COMPLETE!
  Files Generated:
  • brain_tumor_classifier_final.keras  ← best model copy
  • best_brain_tumor_model.keras        ← best checkpoint
  • sample_images.png                   ← sample MRI grid
  • training_curves.png                 ← accuracy/loss graphs
  • confusion_matrix.png                ← per-class performance
═══════════════════════════════════════════════
""")

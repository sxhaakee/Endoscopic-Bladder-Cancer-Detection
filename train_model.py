"""
Bladder Cancer CNN Classifier - Training Script
Downloads the Kaggle dataset and trains a CNN model.
Run this once to generate 'bladder_model.keras'.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub

# ─── CONFIG ────────────────────────────────────────────────────────────────────
IMG_SIZE   = (128, 128)
BATCH      = 32
EPOCHS     = 30
SEED       = 42
MODEL_PATH = "bladder_model.keras"

# ─── 1. DOWNLOAD DATASET ───────────────────────────────────────────────────────
print("==> Downloading dataset from Kaggle …")
dataset_root = kagglehub.dataset_download(
    "aryashah2k/endoscopic-bladder-tissue-classification-dataset"
)
print(f"    Dataset root: {dataset_root}")

# Auto-discover the folder that contains per-class sub-folders
def find_image_dir(root):
    """Walk until we find a directory whose children are all directories (classes)."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden dirs
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        child_dirs  = [d for d in dirnames]
        child_imgs  = [f for f in filenames if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        if child_dirs and not child_imgs:
            # Check that children contain images
            has_images = any(
                any(f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
                    for f in os.listdir(os.path.join(dirpath, cd)))
                for cd in child_dirs
            )
            if has_images:
                return dirpath
    return root

data_dir = find_image_dir(dataset_root)
print(f"    Using data directory: {data_dir}")
classes = sorted([d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')])
print(f"    Classes found: {classes}")

# Map classes to cancer / no-cancer
# Known abbreviations in the Endoscopic Bladder Tissue dataset:
# HGC = High Grade Carcinoma   → Cancer
# LGC = Low Grade Carcinoma    → Cancer
# NST = Normal Squamous Tissue → No Cancer
# NTL = Normal Tissue Lining   → No Cancer
KNOWN_CANCER_CLASSES    = {'hgc', 'lgc'}
KNOWN_NONCANCER_CLASSES = {'nst', 'ntl'}

CANCER_KEYWORDS    = ['cancer', 'cis', 'tumor', 'malignant', 'carcinoma', 'dysplasia', 'polyp', 'abnormal', 'lesion', 'hgc', 'lgc']
NONCANCER_KEYWORDS = ['normal', 'healthy', 'benign', 'non', 'negative', 'nst', 'ntl']

def label_class(name):
    n = name.lower()
    # Exact match for known abbreviations first
    if n in KNOWN_CANCER_CLASSES:
        return 'Cancer'
    if n in KNOWN_NONCANCER_CLASSES:
        return 'No Cancer'
    # Keyword search
    for kw in CANCER_KEYWORDS:
        if kw in n:
            return 'Cancer'
    for kw in NONCANCER_KEYWORDS:
        if kw in n:
            return 'No Cancer'
    return None

mapping = {c: label_class(c) for c in classes}
print(f"    Label mapping: {mapping}")

# If mapping still has None, use fallback but warn
if None in mapping.values():
    print("    ⚠️  Some classes could not be mapped — applying fallback.")
    print("    ⚠️  Please verify the labels below are correct for your dataset!")
    # Last resort: classes with None → label as Cancer (safer for medical use)
    for c in mapping:
        if mapping[c] is None:
            mapping[c] = 'Cancer'
            print(f"    ⚠️  '{c}' defaulted to 'Cancer' — please verify!")
    print(f"    Fallback mapping applied: {mapping}")

# Save class mapping for inference
import json
with open("class_mapping.json", "w") as f:
    json.dump({"classes": classes, "mapping": mapping}, f, indent=2)
print("    class_mapping.json saved.")

# ─── 2. BUILD CUSTOM FLOW WITH RE-LABELLING ───────────────────────────────────
# We'll create a simple custom dataset by symlinking / copying into two folders.
import shutil

def prepare_binary_dataset(data_dir, out_dir, mapping):
    for label in ['Cancer', 'No Cancer']:
        os.makedirs(os.path.join(out_dir, label), exist_ok=True)

    count = {'Cancer': 0, 'No Cancer': 0}
    for cls, lbl in mapping.items():
        if lbl is None:
            continue
        src_folder = os.path.join(data_dir, cls)
        dst_folder = os.path.join(out_dir, lbl)
        for fname in os.listdir(src_folder):
            if fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                src = os.path.join(src_folder, fname)
                # avoid name collisions
                dst_name = f"{cls}_{fname}"
                dst = os.path.join(dst_folder, dst_name)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                count[lbl] += 1
    return count

PREPARED_DIR = "data_binary"
# Always rebuild to ensure correct labels
if os.path.exists(PREPARED_DIR):
    print("==> Removing old binary dataset to rebuild with correct labels …")
    shutil.rmtree(PREPARED_DIR)
if not os.path.exists(PREPARED_DIR):
    print("==> Preparing binary dataset …")
    counts = prepare_binary_dataset(data_dir, PREPARED_DIR, mapping)
    print(f"    Counts: {counts}")
else:
    print("==> Binary dataset already prepared.")
    counts = {
        lbl: len(os.listdir(os.path.join(PREPARED_DIR, lbl)))
        for lbl in ['Cancer', 'No Cancer']
        if os.path.isdir(os.path.join(PREPARED_DIR, lbl))
    }
    print(f"    Counts: {counts}")

# ─── 3. DATA GENERATORS ───────────────────────────────────────────────────────
print("==> Setting up data generators …")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    PREPARED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='binary',
    subset='training',
    seed=SEED,
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    PREPARED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='binary',
    subset='validation',
    seed=SEED,
    shuffle=False
)

# Save class indices for inference
class_indices = train_gen.class_indices   # e.g. {'Cancer': 0, 'No Cancer': 1}
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=2)
print(f"    Class indices: {class_indices}  → saved to class_indices.json")
print(f"    Train samples: {train_gen.samples} | Val samples: {val_gen.samples}")

# ─── 4. BUILD CNN MODEL ────────────────────────────────────────────────────────
print("==> Building CNN model …")

def build_model(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # Dense head
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name='BladderCancerCNN')
    return model

model = build_model()
model.summary()

# Class weights to handle imbalance
total = sum(counts.values())
cancer_count = counts.get('Cancer', 1)
no_cancer_count = counts.get('No Cancer', 1)
# Map to the integer label from class_indices
idx_cancer    = class_indices.get('Cancer', 0)
idx_no_cancer = class_indices.get('No Cancer', 1)
class_weight = {
    idx_cancer:    total / (2.0 * cancer_count),
    idx_no_cancer: total / (2.0 * no_cancer_count)
}
print(f"    Class weights: {class_weight}")

# ─── 5. COMPILE ───────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# ─── 6. CALLBACKS ─────────────────────────────────────────────────────────────
cbs = [
    callbacks.ModelCheckpoint(
        MODEL_PATH, monitor='val_auc', mode='max',
        save_best_only=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4,
        min_lr=1e-6, verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_auc', mode='max',
        patience=8, restore_best_weights=True, verbose=1
    )
]

# ─── 7. TRAIN ─────────────────────────────────────────────────────────────────
print("==> Training …")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=cbs,
    verbose=1
)

# ─── 8. EVALUATE ──────────────────────────────────────────────────────────────
print("\n==> Final evaluation on validation set:")
results = model.evaluate(val_gen, verbose=1)
metrics_names = model.metrics_names
for name, val in zip(metrics_names, results):
    print(f"    {name}: {val:.4f}")

print(f"\n✅ Model saved to: {MODEL_PATH}")
print("   Run `python app.py` to start the web server.")

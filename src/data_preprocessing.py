"""
src/data_preprocessing.py
-------------------------
TensorFlow Data Pipeline and Image Preprocessing for Malaria Detection.

Features:
- Image loading, RGB decoding, 128x128 resizing, and [0, 1] min-max normalization
- Controlled Data Augmentation ONLY on training images:
  * Random Horizontal Flip
  * Random Vertical Flip
  * Small Random Rotation (factor = 0.05, ~18 deg)
  * Small Random Zoom (factor = 0.05)
  * Small Random Contrast adjustment (factor = 0.1)
  * Preserves clinically delicate intracellular ring forms
- Strict avoidance of augmentation on validation and test sets
- High-throughput tf.data pipeline with batching, prefetching (AUTOTUNE), and optional caching
- Single-image preprocessing utility for Streamlit web app and CLI inference
"""

import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import layers


# Preprocessing Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CHANNELS = 3
TARGET_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 64


def build_data_augmentation() -> keras.Sequential:
    """
    Creates a data augmentation sequential block for training only.
    Carefully tuned to avoid distorting delicate parasite chromatin granules.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", name="aug_random_flip"),
            layers.RandomRotation(0.05, fill_mode="nearest", name="aug_random_rotation"),
            layers.RandomZoom(0.05, fill_mode="nearest", name="aug_random_zoom"),
            layers.RandomContrast(0.10, name="aug_random_contrast")
        ],
        name="data_augmentation"
    )


def load_and_preprocess_image(filepath: str, label: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Loads an image from disk, decodes PNG to RGB, resizes to (128, 128), and scales to [0, 1].
    """
    img_bytes = tf.io.read_file(filepath)
    img = tf.io.decode_png(img_bytes, channels=NUM_CHANNELS)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.cast(label, tf.float32)
    return img, label


def create_tf_dataset(
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    is_training: bool = False,
    shuffle_buffer: Optional[int] = None,
    augment: bool = False
) -> tf.data.Dataset:
    """
    Builds a high-performance tf.data.Dataset from a manifest DataFrame.
    
    Args:
        df: DataFrame with 'filepath' and 'label' columns
        batch_size: Number of samples per batch
        is_training: Whether to shuffle
        shuffle_buffer: Buffer size for training shuffle (defaults to len(df) for uniform random mixing)
        augment: Whether to apply clinical-safe data augmentation
    """
    df = df.copy()
    if is_training:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        if shuffle_buffer is None:
            shuffle_buffer = len(df)
            
    filepaths = df["filepath"].values
    labels = df["label"].values.astype(np.float32)
    
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    if is_training:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=42, reshuffle_each_iteration=True)
        
    # Map disk reading and standard preprocessing in parallel
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training and augment:
        aug_block = build_data_augmentation()
        ds = ds.map(
            lambda x, y: (aug_block(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
            
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def load_datasets_from_splits(
    data_dir: str,
    batch_size: int = BATCH_SIZE,
    augment_training: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, validation, and test split CSVs from data_dir and returns tf.data pipelines.
    """
    train_csv = os.path.join(data_dir, "train_split.csv")
    val_csv = os.path.join(data_dir, "val_split.csv")
    test_csv = os.path.join(data_dir, "test_split.csv")
    
    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
        raise FileNotFoundError(
            f"Split files missing in {data_dir}. Run src/data_splitting.py first."
        )
        
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    train_ds = create_tf_dataset(train_df, batch_size=batch_size, is_training=True, augment=augment_training)
    val_ds = create_tf_dataset(val_df, batch_size=batch_size, is_training=False, augment=False)
    test_ds = create_tf_dataset(test_df, batch_size=batch_size, is_training=False, augment=False)
    
    return train_ds, val_ds, test_ds, train_df, val_df, test_df


def preprocess_single_image(
    image_input,
    target_size: Tuple[int, int] = TARGET_SIZE
) -> np.ndarray:
    """
    Preprocesses a single image for model inference.
    Accepts a filepath, PIL Image, or numpy array.
    Returns a batch tensor of shape (1, 128, 128, 3) normalized to [0, 1].
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    else:
        raise ValueError("Unsupported image input type.")
        
    pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # Shape: (1, 128, 128, 3)
    return arr


if __name__ == "__main__":
    split_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    tr_ds, va_ds, te_ds, tr_df, va_df, te_df = load_datasets_from_splits(split_dir, batch_size=32)
    for x, y in tr_ds.take(1):
        print("Sample batch X shape:", x.shape, "Y shape:", y.shape)
        print("Batch X range:", tf.reduce_min(x).numpy(), "to", tf.reduce_max(x).numpy())

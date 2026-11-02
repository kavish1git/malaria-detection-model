"""
src/model.py
------------
Custom Convolutional Neural Network (CNN) Architecture for Malaria Detection.

Design Specifications:
- Input Shape: (128, 128, 3)
- Feature Extraction: 4 Convolutional blocks with increasing filters (32, 64, 128, 256)
- Regularization:
    * Batch Normalization after every convolution to stabilize training dynamics and accelerate convergence
    * Progressively increasing Spatial Dropout (0.20 -> 0.25 -> 0.30)
    * Global Average Pooling 2D to drastically reduce parameter count and eliminate spatial overfitting
    * Dense layer with L2 weight decay (kernel_regularizer=l2(1e-4))
    * Heavy Dropout (0.50) prior to final classification
- Output: Single sigmoid unit producing P(Parasitized | Image)
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers


def build_malaria_cnn(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    l2_reg: float = 1e-4,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Constructs and compiles the Malaria CNN model according to research specifications.
    
    Args:
        input_shape: Dimensions of input tensor (height, width, channels)
        l2_reg: L2 weight penalty applied to dense layers
        learning_rate: Initial learning rate for the Adam optimizer
        
    Returns:
        Compiled Keras Model instance
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # -------------------------------------------------------------
    # Block 1: Low-level features (edges, color boundaries, texture)
    # -------------------------------------------------------------
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="block1_conv"
    )(inputs)
    x = layers.BatchNormalization(name="block1_bn")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="block1_pool")(x)
    x = layers.Dropout(0.20, name="block1_dropout")(x)
    
    # -------------------------------------------------------------
    # Block 2: Intermediate features (cell contours, cytoplasm boundaries)
    # -------------------------------------------------------------
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="block2_conv"
    )(x)
    x = layers.BatchNormalization(name="block2_bn")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="block2_pool")(x)
    x = layers.Dropout(0.25, name="block2_dropout")(x)
    
    # -------------------------------------------------------------
    # Block 3: High-level features (intracellular inclusions, chromatin spots)
    # -------------------------------------------------------------
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="block3_conv"
    )(x)
    x = layers.BatchNormalization(name="block3_bn")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="block3_pool")(x)
    x = layers.Dropout(0.30, name="block3_dropout")(x)
    
    # -------------------------------------------------------------
    # Block 4: Complex morphological patterns (malaria ring forms, trophozoites)
    # -------------------------------------------------------------
    x = layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="block4_conv"
    )(x)
    x = layers.BatchNormalization(name="block4_bn")(x)
    
    # Global Average Pooling replaces flattening:
    # Aggregates feature maps across spatial dimensions into 1D 256-dim embedding.
    # Drastically suppresses overfitting by removing dense spatial weights.
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    # -------------------------------------------------------------
    # Dense Classification Head
    # -------------------------------------------------------------
    x = layers.Dense(
        units=128,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_128"
    )(x)
    x = layers.BatchNormalization(name="dense_bn")(x)
    x = layers.Dropout(0.50, name="dense_dropout")(x)
    
    outputs = layers.Dense(
        units=1,
        activation="sigmoid",
        name="output_probability"
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Malaria_Detection_CNN")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR")
        ]
    )
    
    return model


if __name__ == "__main__":
    model = build_malaria_cnn()
    model.summary()

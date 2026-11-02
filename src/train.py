"""
src/train.py
------------
Full Training, Overfitting Prevention, and Automated Evaluation Orchestration.

Features:
- Configurable Hyperparameters (Batch size, Learning rate, L2 regularization, Epochs)
- Overfitting Prevention Suite:
  * Safe Clinical Data Augmentation
  * Dropout (20%, 25%, 30%, 50%)
  * L2 Weight Regularization
  * Batch Normalization
  * EarlyStopping with restore_best_weights=True (patience=7)
  * ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6)
  * ModelCheckpoint saving best validation model
- End-to-end evaluation on untouched test set with all requested clinical metrics
- Automatic generation of all required academic plots
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.data_preprocessing import load_datasets_from_splits, BATCH_SIZE
from src.model import build_malaria_cnn
from src.evaluate import (
    calculate_clinical_metrics, find_optimal_threshold_val,
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_threshold_analysis, plot_reliability_diagram,
    plot_sample_predictions, plot_training_history
)


def train_malaria_model(
    data_dir: str = os.path.join(PROJECT_ROOT, "data"),
    models_dir: str = os.path.join(PROJECT_ROOT, "models"),
    results_dir: str = os.path.join(PROJECT_ROOT, "results"),
    epochs: int = 40,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = 0.001,
    patience: int = 7
) -> Dict[str, Any]:
    """
    Executes model training, checkpointing, and evaluation.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, "malaria_cnn.keras")
    
    print("=" * 65)
    print("STEP 1: Loading tf.data pipelines from patient-grouped splits")
    print("=" * 65)
    train_ds, val_ds, test_ds, train_df, val_df, test_df = load_datasets_from_splits(
        data_dir=data_dir,
        batch_size=batch_size,
        augment_training=True
    )
    print(f"Training samples:   {len(train_df):,} across {train_df['patient_id'].nunique()} patients")
    print(f"Validation samples: {len(val_df):,} across {val_df['patient_id'].nunique()} patients")
    print(f"Testing samples:    {len(test_df):,} across {test_df['patient_id'].nunique()} patients")
    
    print("\n" + "=" * 65)
    print("STEP 2: Initializing Custom CNN Architecture")
    print("=" * 65)
    model = build_malaria_cnn(
        input_shape=(128, 128, 3),
        l2_reg=1e-4,
        learning_rate=learning_rate
    )
    model.summary()
    
    # Callbacks for robust overfitting mitigation
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, "training_log.csv"),
            separator=",",
            append=False
        )
    ]
    
    print("\n" + "=" * 65)
    print(f"STEP 3: Initiating Model Training (Max {epochs} epochs)")
    print("=" * 65)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history dictionary
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        # Convert float32 to float for json serialization
        hist_serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(hist_serializable, f, indent=2)
        
    # Generate Training History Curve
    history_plot_path = os.path.join(results_dir, "training_history.png")
    plot_training_history(history.history, history_plot_path)
    print(f"Training history visualization saved to: {history_plot_path}")
    
    # Load best checkpointed model to ensure exact best weights
    if os.path.exists(model_save_path):
        print(f"Loading best saved weights from {model_save_path}")
        best_model = keras.models.load_model(model_save_path)
    else:
        best_model = model
        
    print("\n" + "=" * 65)
    print("STEP 4: Validation Set Analysis & Optimal Threshold Discovery")
    print("=" * 65)
    print("Generating validation probabilities...")
    y_val_prob = best_model.predict(val_ds, verbose=1).ravel()
    y_val_true = val_df["label"].values.astype(int)
    
    opt_th, opt_val_f1 = find_optimal_threshold_val(y_val_true, y_val_prob)
    print(f"Default Decision Threshold: 0.50")
    print(f"Validation-Optimal Decision Threshold: {opt_th:.3f} (Val F1: {opt_val_f1:.4f})")
    
    # Save validation threshold plot
    th_plot_path = os.path.join(results_dir, "threshold_analysis.png")
    plot_threshold_analysis(y_val_true, y_val_prob, opt_th, th_plot_path)
    
    print("\n" + "=" * 65)
    print("STEP 5: Comprehensive Evaluation on UNTOUCHED Test Dataset")
    print("=" * 65)
    print("Generating test probabilities...")
    y_test_prob = best_model.predict(test_ds, verbose=1).ravel()
    y_test_true = test_df["label"].values.astype(int)
    
    # Evaluate at default threshold (0.50)
    metrics_default = calculate_clinical_metrics(y_test_true, y_test_prob, threshold=0.50)
    # Evaluate at validation-selected threshold (opt_th)
    metrics_optimal = calculate_clinical_metrics(y_test_true, y_test_prob, threshold=opt_th)
    
    metrics_summary_df = pd.DataFrame([
        {"Condition": "Default Threshold (0.50)", **metrics_default},
        {"Condition": f"Val-Tuned Threshold ({opt_th:.2f})", **metrics_optimal}
    ])
    
    metrics_csv_path = os.path.join(results_dir, "evaluation_metrics.csv")
    metrics_summary_df.to_csv(metrics_csv_path, index=False)
    
    print("\nTest Evaluation Metrics Summary:")
    print(metrics_summary_df.T.to_string())
    
    # Classification Report
    y_test_pred_def = (y_test_prob >= 0.50).astype(int)
    from sklearn.metrics import classification_report
    cr_dict = classification_report(
        y_test_true, y_test_pred_def,
        target_names=["Uninfected (0)", "Parasitized (1)"],
        output_dict=True
    )
    cr_df = pd.DataFrame(cr_dict).transpose()
    cr_csv_path = os.path.join(results_dir, "classification_report.csv")
    cr_df.to_csv(cr_csv_path)
    print("\nClassification Report (Test Set):")
    print(cr_df.to_string())
    
    print("\n" + "=" * 65)
    print("STEP 6: Generating Academic Figures for Presentation")
    print("=" * 65)
    plot_confusion_matrix(
        y_test_true, y_test_pred_def,
        save_path=os.path.join(results_dir, "confusion_matrix.png")
    )
    plot_roc_curve(
        y_test_true, y_test_prob,
        save_path=os.path.join(results_dir, "roc_curve.png")
    )
    plot_precision_recall_curve(
        y_test_true, y_test_prob,
        save_path=os.path.join(results_dir, "precision_recall_curve.png")
    )
    plot_reliability_diagram(
        y_test_true, y_test_prob,
        save_path=os.path.join(results_dir, "reliability_diagram.png")
    )
    plot_sample_predictions(
        test_df, y_test_prob,
        save_path=os.path.join(results_dir, "sample_predictions.png"),
        threshold=0.50
    )
    
    print("All visualizations generated successfully in:", results_dir)
    print("Model successfully preserved in:", model_save_path)
    print("=" * 65)
    
    return {
        "model_path": model_save_path,
        "metrics_default": metrics_default,
        "metrics_optimal": metrics_optimal,
        "optimal_threshold": opt_th
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Malaria Detection CNN")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    args = parser.parse_args()
    
    train_malaria_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience
    )

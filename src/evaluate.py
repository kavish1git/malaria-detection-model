"""
src/evaluate.py
---------------
Comprehensive Model Evaluation and Academic Visualization Suite.

Calculates:
1. Basic Classification Metrics:
   - Accuracy, Precision (PPV), Recall (Sensitivity), Specificity,
   - F1-score, F2-score, Balanced Accuracy, Negative Predictive Value (NPV)
2. Advanced Classification Metrics:
   - ROC-AUC, PR-AUC (Average Precision), Matthews Correlation Coefficient (MCC),
   - Cohen's Kappa, Log Loss, Brier Score, Expected Calibration Error (ECE)
3. Confusion Matrix (TP, TN, FP, FN)
4. Classification Report
5. Optimal Threshold Discovery (Optimized strictly on VALIDATION set, applied to TEST set)
6. Academic Presentation Figures:
   - Confusion Matrix Heatmap
   - ROC Curve
   - Precision-Recall Curve
   - Threshold vs F1 / Precision / Recall Sweep
   - Calibration Curve (Reliability Diagram)
   - Sample Predictions Grid (TP, TN, FP, FN error analysis)
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

# Set presentation-grade style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, float]:
    """
    Computes all standard and clinical diagnostic metrics for binary malaria detection.
    Positive Class: Parasitized (1)
    Negative Class: Uninfected (0)
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0) # PPV
    rec = recall_score(y_true, y_pred, zero_division=0)     # Sensitivity
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0          # Specificity
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0           # Negative Predictive Value
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    # Advanced metrics
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.0
        
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0
        
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    ce_loss = log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        "Threshold": threshold,
        "Accuracy": acc,
        "Precision (PPV)": prec,
        "Recall / Sensitivity": rec,
        "Specificity (TNR)": spec,
        "Negative Predictive Value (NPV)": npv,
        "F1-Score": f1,
        "F2-Score": f2,
        "Balanced Accuracy": bacc,
        "ROC-AUC": roc_auc,
        "PR-AUC (Avg Precision)": pr_auc,
        "Matthews Corr Coef (MCC)": mcc,
        "Cohen's Kappa": kappa,
        "Log Loss (Cross-Entropy)": ce_loss,
        "Brier Score": brier,
        "True Positives (TP)": int(tp),
        "True Negatives (TN)": int(tn),
        "False Positives (FP)": int(fp),
        "False Negatives (FN)": int(fn)
    }


def find_optimal_threshold_val(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray
) -> Tuple[float, float]:
    """
    Finds the optimal decision threshold that maximizes F1-score on the VALIDATION set.
    Prevents test dataset snooping.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_f1 = -1.0
    best_th = 0.50
    
    for th in thresholds:
        preds = (y_val_prob >= th).astype(int)
        score = f1_score(y_val_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = th
            
    return float(best_th), float(best_f1)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix - Test Dataset"
):
    """
    Generates and saves a publication-grade annotated confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    cm_labels = [
        [f"True Negative\n(Uninfected)\n{tn:,}\n({tn/len(y_true)*100:.1f}%)",
         f"False Positive\n(Type I Error)\n{fp:,}\n({fp/len(y_true)*100:.1f}%)"],
        [f"False Negative\n(Type II Error - Missed!)\n{fn:,}\n({fn/len(y_true)*100:.1f}%)",
         f"True Positive\n(Parasitized)\n{tp:,}\n({tp/len(y_true)*100:.1f}%)"]
    ]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=np.array(cm_labels),
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Uninfected", "Predicted Parasitized"],
        yticklabels=["Actual Uninfected", "Actual Parasitized"],
        annot_kws={"size": 11, "weight": "bold"},
        ax=ax
    )
    ax.set_title(title, fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str
):
    """
    Plots the Receiver Operating Characteristic (ROC) curve with AUC annotation.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Malaria CNN (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=10)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, labelpad=10)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, weight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str
):
    """
    Plots the Precision-Recall curve with Average Precision (PR-AUC) annotation.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    no_skill = np.sum(y_true) / len(y_true)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#2ca02c", lw=2.5, label=f"Malaria CNN (AP = {ap_score:.4f})")
    ax.plot([0, 1], [no_skill, no_skill], color="#7f7f7f", lw=1.5, linestyle="--", label=f"Baseline Baseline (AP = {no_skill:.4f})")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, labelpad=10)
    ax.set_ylabel("Precision (PPV)", fontsize=11, labelpad=10)
    ax.set_title("Precision-Recall Curve", fontsize=13, weight="bold", pad=15)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_threshold_analysis(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    optimal_th: float,
    save_path: str
):
    """
    Plots F1, Precision, and Recall across candidate decision thresholds.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    f1s, precs, recs = [], [], []
    
    for th in thresholds:
        preds = (y_val_prob >= th).astype(int)
        f1s.append(f1_score(y_val_true, preds, zero_division=0))
        precs.append(precision_score(y_val_true, preds, zero_division=0))
        recs.append(recall_score(y_val_true, preds, zero_division=0))
        
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(thresholds, f1s, label="F1-Score", color="#d62728", lw=2.2)
    ax.plot(thresholds, precs, label="Precision (PPV)", color="#1f77b4", lw=2.0, linestyle="--")
    ax.plot(thresholds, recs, label="Recall (Sensitivity)", color="#2ca02c", lw=2.0, linestyle="-.")
    
    ax.axvline(0.50, color="#555555", linestyle=":", lw=1.5, label="Default Threshold (0.50)")
    ax.axvline(optimal_th, color="#e377c2", linestyle="-", lw=2.0, label=f"Optimal Val Threshold ({optimal_th:.2f})")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Classification Decision Threshold", fontsize=11, labelpad=10)
    ax.set_ylabel("Metric Value", fontsize=11, labelpad=10)
    ax.set_title("Trade-off Analysis: Threshold vs. F1, Precision & Recall (Validation)", fontsize=13, weight="bold", pad=15)
    ax.legend(loc="lower center", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str
):
    """
    Plots the calibration curve / reliability diagram with Brier Score and confidence histogram.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [2.5, 1]}, sharex=True)
    
    # Calibration plot
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax1.plot(prob_pred, prob_true, marker="o", color="#ff7f0e", lw=2.5, label=f"CNN Model (Brier = {brier:.4f})")
    ax1.set_ylabel("Fraction of Positives (Empirical)", fontsize=11)
    ax1.set_title("Model Calibration & Reliability Diagram", fontsize=13, weight="bold", pad=12)
    ax1.legend(loc="upper left", frameon=True)
    
    # Probability confidence histogram
    ax2.hist(y_prob, bins=20, range=(0, 1), color="#4c72b0", edgecolor="black", alpha=0.75)
    ax2.set_xlabel("Predicted Probability (Confidence)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_sample_predictions(
    test_df: pd.DataFrame,
    y_prob: np.ndarray,
    save_path: str,
    threshold: float = 0.50
):
    """
    Displays representative examples of TP, TN, FP, and FN from the test set for clinical error analysis.
    """
    y_true = test_df["label"].values
    y_pred = (y_prob >= threshold).astype(int)
    
    tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    tn_indices = np.where((y_true == 0) & (y_pred == 0))[0]
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    
    samples_to_show = [
        ("True Positive (Correct Parasitized)", tp_indices[:2], "#2ca02c"),
        ("True Negative (Correct Uninfected)", tn_indices[:2], "#1f77b4"),
        ("False Positive (Type I: False Alarm)", fp_indices[:2], "#d62728"),
        ("False Negative (Type II: Missed Infection)", fn_indices[:2], "#9467bd")
    ]
    
    fig, axes = plt.subplots(4, 2, figsize=(8, 12))
    row = 0
    for title, indices, border_color in samples_to_show:
        for col_idx, idx in enumerate(indices):
            ax = axes[row, col_idx]
            if idx < len(test_df):
                fpath = test_df.iloc[idx]["filepath"]
                prob = y_prob[idx]
                patient = test_df.iloc[idx]["patient_id"]
                try:
                    img = Image.open(fpath)
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Image Load Error", ha="center")
                ax.set_title(
                    f"{title}\nP(Infected)={prob:.3f} | {patient}",
                    fontsize=9, weight="bold", color=border_color
                )
            ax.axis("off")
        row += 1
        
    plt.suptitle("Clinical Error Analysis: Test Prediction Gallery", fontsize=14, weight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_history(history_dict: Dict[str, list], save_path: str):
    """
    Plots training vs validation accuracy and loss over epochs.
    """
    epochs = range(1, len(history_dict["loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Loss plot
    ax1.plot(epochs, history_dict["loss"], label="Training Loss", color="#1f77b4", lw=2.2)
    ax1.plot(epochs, history_dict["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2.2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Training Loss vs. Validation Loss", fontsize=13, weight="bold", pad=12)
    ax1.legend(frameon=True)
    
    # Accuracy plot
    ax2.plot(epochs, history_dict["accuracy"], label="Training Accuracy", color="#2ca02c", lw=2.2)
    ax2.plot(epochs, history_dict["val_accuracy"], label="Validation Accuracy", color="#d62728", lw=2.2)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Training Accuracy vs. Validation Accuracy", fontsize=13, weight="bold", pad=12)
    ax2.legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

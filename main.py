"""
main.py
-------
Main Execution Pipeline for Malaria Detection Deep Learning System.

Orchestrates the entire lifecycle:
1. Verifies raw dataset availability and integrity
2. Executes strict patient-level grouped data splitting (Zero Patient Leakage)
3. Trains the 4-block custom CNN with multi-tier regularization (Dropout, BatchNorm, L2, GAP, Augmentation)
4. Employs EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau
5. Conducts comprehensive diagnostic evaluation on untouched test set
6. Generates publication-ready figures in results/
7. Prepares the trained model for Streamlit deployment
"""

import os
import sys
import argparse

# Set Project Root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_splitting import generate_and_save_splits
from src.train import train_malaria_model


def run_pipeline(
    raw_data_dir: str = os.path.join(PROJECT_ROOT, "data", "cell_images"),
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 7
):
    print("=" * 70)
    print("MALARIA CELL CLASSIFICATION: CONVOLUTIONAL NEURAL NETWORK PIPELINE")
    print("=" * 70)
    
    data_dir = os.path.join(PROJECT_ROOT, "data")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    
    # 1. Dataset Verification & Splitting
    split_manifest = os.path.join(data_dir, "dataset_manifest.csv")
    if not os.path.exists(split_manifest):
        print(f"\n[PHASE 1] Creating patient-level grouped splits from: {raw_data_dir}")
        if not os.path.exists(raw_data_dir):
            raise FileNotFoundError(f"Raw dataset path does not exist: {raw_data_dir}")
        generate_and_save_splits(raw_data_dir, data_dir)
    else:
        print(f"\n[PHASE 1] Existing patient-level split manifests found in: {data_dir}")
        
    # 2. Training and Evaluation
    print(f"\n[PHASE 2] Commencing CNN Model Training & Academic Evaluation")
    results = train_malaria_model(
        data_dir=data_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience
    )
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Saved Trained Model : {results['model_path']}")
    print(f"Results Directory   : {results_dir}")
    print("To launch the interactive web demo, run:")
    print("    streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Complete Malaria AI Pipeline")
    parser.add_argument("--data_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "cell_images"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    
    run_pipeline(
        raw_data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience
    )

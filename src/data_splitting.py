"""
src/data_splitting.py
---------------------
Patient-Level Group-Based Data Splitting to Prevent Data Leakage.

Context & Scientific Rationale:
The NIH Malaria Cell Images Dataset originates from 150 malaria-infected patients
and 50 healthy non-infected individuals (total 200 slide/patient specimens).
Filenames encode patient identifiers (e.g., C100P61... -> Patient P61; C12N... -> Control C12N).

If a naive random image-level split is used:
- Multiple erythrocyte crops from the exact same microscope slide/patient are scattered
  across training, validation, and testing sets.
- The CNN may learn slide-specific artifacts (Giemsa stain concentration, lighting,
  sensor noise, patient-specific RBC morphology) rather than genuine plasmodium parasites.
- This creates severe data leakage, artificially inflating validation/test accuracy and
  leading to disastrously poor generalization in real-world clinical deployment.

This module guarantees strict patient-level grouped splitting:
- Training: ~70%
- Validation: ~15%
- Testing: ~15%
- Stratified across infected patients and healthy control donors.
- ZERO patient overlap across splits:
  intersection(Train_Patients, Val_Patients) = empty
  intersection(Train_Patients, Test_Patients) = empty
  intersection(Val_Patients, Test_Patients) = empty
"""

import os
import re
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd


def extract_patient_id(filename: str) -> Tuple[str, str]:
    """
    Extracts the patient/donor group identifier and group type from filename.
    
    Examples:
      - C100P61ThinF_IMG_... -> Patient_P61, 'infected_patient'
      - C12NThinF_IMG_...   -> Control_C12N, 'control'
    """
    m_patient = re.search(r"P(\d+)", filename)
    m_control = re.search(r"(C\d+N)", filename)
    
    if m_patient:
        return f"Patient_P{m_patient.group(1)}", "infected_patient"
    elif m_control:
        return f"Control_{m_control.group(1)}", "control"
    else:
        # Fallback to prefix before ThinF or first underscore
        prefix = filename.split("ThinF")[0].split("thinF")[0].split("_")[0]
        return f"Group_{prefix}", "other"


def create_dataset_manifest(
    data_dir: str,
    categories: Tuple[str, ...] = ("Parasitized", "Uninfected")
) -> pd.DataFrame:
    """
    Scans the dataset folder, filters out non-image files, extracts patient metadata,
    and returns a master pandas DataFrame.
    """
    records = []
    for cat in categories:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.exists(cat_dir):
            raise FileNotFoundError(f"Category folder not found: {cat_dir}")
            
        for fname in os.listdir(cat_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                # Skip Thumbs.db and non-image metadata
                continue
                
            fpath = os.path.join(cat_dir, fname)
            patient_id, group_type = extract_patient_id(fname)
            label = 1 if cat == "Parasitized" else 0
            
            records.append({
                "filename": fname,
                "category": cat,
                "label": label,
                "patient_id": patient_id,
                "group_type": group_type,
                "filepath": fpath
            })
            
    df = pd.DataFrame(records)
    return df


def split_patient_groups(
    patient_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits a list of unique patient IDs into train, validation, and test subsets.
    """
    rng = np.random.RandomState(random_seed)
    shuffled = sorted(list(set(patient_ids)))
    rng.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    
    train_pids = shuffled[:n_train]
    val_pids = shuffled[n_train:n_train + n_val]
    test_pids = shuffled[n_train + n_val:]
    
    return train_pids, val_pids, test_pids


def perform_patient_level_split(
    manifest_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Performs stratified group splitting at the patient level.
    Ensures that infected patients and control donors are partitioned
    proportionally across train, val, and test splits without any patient overlap.
    """
    df = manifest_df.copy()
    
    # Stratify by group type (infected_patient vs control)
    infected_pids = df[df["group_type"] == "infected_patient"]["patient_id"].unique()
    control_pids = df[df["group_type"] == "control"]["patient_id"].unique()
    other_pids = df[df["group_type"] == "other"]["patient_id"].unique()
    
    inf_tr, inf_val, inf_te = split_patient_groups(infected_pids, train_ratio, val_ratio, random_seed)
    ctl_tr, ctl_val, ctl_te = split_patient_groups(control_pids, train_ratio, val_ratio, random_seed)
    oth_tr, oth_val, oth_te = split_patient_groups(other_pids, train_ratio, val_ratio, random_seed)
    
    train_patients = set(inf_tr + ctl_tr + oth_tr)
    val_patients = set(inf_val + ctl_val + oth_val)
    test_patients = set(inf_te + ctl_te + oth_te)
    
    # Assert zero patient leakage
    assert len(train_patients & val_patients) == 0, "Patient leakage detected between train and val!"
    assert len(train_patients & test_patients) == 0, "Patient leakage detected between train and test!"
    assert len(val_patients & test_patients) == 0, "Patient leakage detected between val and test!"
    
    df["split"] = "unknown"
    df.loc[df["patient_id"].isin(train_patients), "split"] = "train"
    df.loc[df["patient_id"].isin(val_patients), "split"] = "val"
    df.loc[df["patient_id"].isin(test_patients), "split"] = "test"
    
    return df


def generate_and_save_splits(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Executes manifest creation, stratified patient splitting, verification,
    and saves split CSV files into data/ directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest = create_dataset_manifest(data_dir)
    split_df = perform_patient_level_split(
        manifest,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )
    
    # Save master manifest and individual split CSVs
    manifest_path = os.path.join(output_dir, "dataset_manifest.csv")
    split_df.to_csv(manifest_path, index=False)
    
    train_df = split_df[split_df["split"] == "train"].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    val_df = split_df[split_df["split"] == "val"].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    test_df = split_df[split_df["split"] == "test"].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")
    test_path = os.path.join(output_dir, "test_split.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("=" * 60)
    print("PATIENT-LEVEL DATA SPLITTING SUMMARY")
    print("=" * 60)
    print(f"Total Images: {len(split_df):,}")
    print(f"Total Unique Patients: {split_df['patient_id'].nunique()}")
    print("-" * 60)
    
    for split_name, sub_df in [("Training", train_df), ("Validation", val_df), ("Testing", test_df)]:
        n_tot = len(sub_df)
        n_parasitized = (sub_df["label"] == 1).sum()
        n_uninfected = (sub_df["label"] == 0).sum()
        n_patients = sub_df["patient_id"].nunique()
        pct = (n_tot / len(split_df)) * 100
        print(f"{split_name:12s}: {n_tot:5d} images ({pct:5.1f}%) | "
              f"Parasitized: {n_parasitized:5d} | "
              f"Uninfected: {n_uninfected:5d} | "
              f"Patients: {n_patients:3d}")
              
    print("-" * 60)
    print("Patient Leakage Verification:")
    p_tr = set(train_df["patient_id"])
    p_va = set(val_df["patient_id"])
    p_te = set(test_df["patient_id"])
    print(f"Train / Val  Patient Overlap: {len(p_tr & p_va)} (Clean)")
    print(f"Train / Test Patient Overlap: {len(p_tr & p_te)} (Clean)")
    print(f"Val   / Test Patient Overlap: {len(p_va & p_te)} (Clean)")
    print("=" * 60)
    print(f"Saved split files to: {output_dir}")
    
    return {
        "manifest": split_df,
        "train": train_df,
        "val": val_df,
        "test": test_df
    }


if __name__ == "__main__":
    _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(_proj, "data", "cell_images")
    OUT_DIR = os.path.join(_proj, "data")
    generate_and_save_splits(DATA_PATH, OUT_DIR)

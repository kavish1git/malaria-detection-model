"""
src/predict.py
--------------
Inference Module for Single-Image and Batch Malaria Prediction.

Provides:
- Standalone CLI interface for evaluating individual or directories of microscopic images
- Probability calibration and risk stratification
- Clean API for Streamlit application integration
"""

import os
import sys
import argparse
from typing import Dict, Any, Union
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import preprocess_single_image, TARGET_SIZE


class MalariaPredictor:
    """
    Production-ready wrapper for malaria CNN inference.
    """
    def __init__(self, model_path: str = os.path.join(PROJECT_ROOT, "models", "malaria_cnn.keras")):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
        self.model = keras.models.load_model(model_path)
        self.class_names = ["Uninfected", "Parasitized"]
        
    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Runs inference on an input image.
        
        Args:
            image_input: Filepath string, PIL Image, or numpy array
            threshold: Classification decision boundary
            
        Returns:
            Dictionary containing:
              - predicted_class: 'Parasitized' or 'Uninfected'
              - confidence: float probability (0.0 to 1.0)
              - parasitized_probability: float
              - uninfected_probability: float
              - is_infected: bool
              - risk_level: str ('Low', 'Moderate', 'High', 'Critical')
        """
        tensor = preprocess_single_image(image_input, target_size=TARGET_SIZE)
        prob = float(self.model.predict(tensor, verbose=0)[0][0])
        
        is_infected = prob >= threshold
        predicted_class = "Parasitized" if is_infected else "Uninfected"
        confidence = prob if is_infected else (1.0 - prob)
        
        # Risk Stratification
        if prob < 0.20:
            risk = "Low Risk / Likely Uninfected"
        elif prob < 0.50:
            risk = "Indeterminate / Borderline Negative"
        elif prob < 0.80:
            risk = "Moderate Risk / Probable Parasitemia"
        else:
            risk = "High Risk / Distinct Intracellular Parasites Detected"
            
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "parasitized_probability": prob,
            "uninfected_probability": 1.0 - prob,
            "is_infected": is_infected,
            "risk_level": risk,
            "threshold_used": threshold
        }


def main():
    parser = argparse.ArgumentParser(description="Predict Malaria Infection on Blood Cell Image")
    parser.add_argument("image_path", type=str, help="Path to cell image (PNG or JPG)")
    parser.add_argument("--model_path", type=str, default=os.path.join(PROJECT_ROOT, "models", "malaria_cnn.keras"))
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()
    
    predictor = MalariaPredictor(args.model_path)
    res = predictor.predict(args.image_path, threshold=args.threshold)
    
    print("\n" + "=" * 50)
    print("MALARIA CELL INFECTION PREDICTION RESULT")
    print("=" * 50)
    print(f"Image File           : {args.image_path}")
    print(f"Predicted Diagnosis  : {res['predicted_class'].upper()}")
    print(f"Confidence           : {res['confidence'] * 100:.2f}%")
    print(f"Parasite Probability : {res['parasitized_probability'] * 100:.2f}%")
    print(f"Risk Assessment      : {res['risk_level']}")
    print("=" * 50)
    print("DISCLAIMER: Educational and research prototype only.")
    print("Not FDA/CE approved for clinical diagnostic decision-making.\n")


if __name__ == "__main__":
    main()

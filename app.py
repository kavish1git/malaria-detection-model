"""
app.py
------
Streamlit Web Application: Microscopic Malaria Detection Assistant.

Provides an interactive user interface for:
- Uploading blood cell microscopic images (PNG, JPG)
- Selecting from pre-loaded demonstration test samples
- Real-time deep learning inference using the custom Malaria CNN
- Confidence visualization, class probability breakdown, and risk stratification
- Comprehensive medical and research disclaimer
"""

import os
import sys
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is accessible
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import MalariaPredictor

# Page Configuration
st.set_page_config(
    page_title="Malaria Detection AI | Blood Smear Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .disclaimer-card {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.88rem;
        margin-bottom: 1.5rem;
    }
    .result-box-parasitized {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        margin-top: 10px;
    }
    .result-box-uninfected {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    model_path = os.path.join(PROJECT_ROOT, "models", "malaria_cnn.keras")
    if not os.path.exists(model_path):
        return None
    return MalariaPredictor(model_path=model_path)


def main():
    st.markdown('<div class="main-header">🔬 Microscopic Malaria Detection AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Deep Learning-Assisted Giemsa-Stained Erythrocyte Classification | NIH Malaria Dataset</div>',
        unsafe_allow_html=True
    )

    # Educational Disclaimer Banner
    st.markdown("""
    <div class="disclaimer-card">
        <strong>⚠️ CLINICAL DISCLAIMER:</strong> This application is an <em>academic and educational research prototype</em>
        developed for demonstrative purposes. It has <strong>not</strong> been evaluated by medical regulatory authorities
        (FDA, CE) and must <strong>never</strong> be used as a substitute for professional hematological examination,
        pathologist microscopy, or clinical diagnostic decision-making.
    </div>
    """, unsafe_allow_html=True)

    predictor = load_predictor()

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("Adjust inference parameters and explore model architecture details.")
        
        threshold = st.slider(
            "Classification Decision Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Threshold for labeling as Parasitized. Lower threshold increases Sensitivity (Recall) to reduce missed infections."
        )
        
        st.markdown("---")
        st.subheader("Model Specifications")
        st.markdown("""
        - **Architecture:** 4-Block Custom CNN
        - **Input Resolution:** 128 × 128 × 3 RGB
        - **Regularization:** BatchNorm, Spatial Dropout (20-30%), Dense Dropout (50%), GAP, L2
        - **Training Split:** Strict Patient-Level Group Split (Zero Patient Leakage)
        """)

    if predictor is None:
        st.error(
            "⚠️ Trained model not found at `models/malaria_cnn.keras`. "
            "Please train the model first by running `python main.py` or `python src/train.py`."
        )
        return

    # Main Area: Two Columns
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Input Image Selection")
        input_mode = st.radio(
            "Choose Input Source:",
            ["Upload Your Own Cell Image", "Select Demonstration Sample"],
            horizontal=True
        )

        image_to_process = None
        img_name = ""

        if input_mode == "Upload Your Own Cell Image":
            uploaded_file = st.file_uploader(
                "Upload a thin blood smear cell crop (PNG or JPG):",
                type=["png", "jpg", "jpeg"]
            )
            if uploaded_file is not None:
                image_to_process = Image.open(uploaded_file)
                img_name = uploaded_file.name
        else:
            # Demo images from test set manifest if available
            test_csv = os.path.join(PROJECT_ROOT, "data", "test_split.csv")
            if os.path.exists(test_csv):
                test_df = pd.read_csv(test_csv)
                para_sample = test_df[test_df["category"] == "Parasitized"].iloc[0]["filepath"]
                uninf_sample = test_df[test_df["category"] == "Uninfected"].iloc[0]["filepath"]
                
                demo_choice = st.selectbox(
                    "Select an untouched test-set sample:",
                    [
                        "Malaria Parasitized Cell (Test Set)",
                        "Uninfected Healthy Erythrocyte (Test Set)"
                    ]
                )
                sample_path = para_sample if "Parasitized" in demo_choice else uninf_sample
                if os.path.exists(sample_path):
                    image_to_process = Image.open(sample_path)
                    img_name = os.path.basename(sample_path)
            else:
                st.info("Test split manifest not detected. Please upload an image directly.")

        if image_to_process is not None:
            st.image(image_to_process, caption=f"Selected Cell: {img_name}", width=250)
            orig_w, orig_h = image_to_process.size
            st.caption(f"Original Dimensions: {orig_w} × {orig_h} | Resized automatically to 128 × 128 for CNN inference.")

    with col2:
        st.subheader("2. AI Diagnostic Inference")

        if image_to_process is not None:
            with st.spinner("Analyzing microscopic morphology..."):
                result = predictor.predict(image_to_process, threshold=threshold)

            is_infected = result["is_infected"]
            prob_para = result["parasitized_probability"]
            conf = result["confidence"]
            pred_class = result["predicted_class"]

            if is_infected:
                st.markdown(f"""
                <div class="result-box-parasitized">
                    <h2 style="color: #DC2626; margin: 0;">🦠 PARASITIZED DETECTED</h2>
                    <p style="font-size: 1.1rem; color: #7F1D1D; margin-top: 5px;">
                        Confidence: <strong>{conf * 100:.2f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box-uninfected">
                    <h2 style="color: #059669; margin: 0;">✅ UNINFECTED ERYTHROCYTE</h2>
                    <p style="font-size: 1.1rem; color: #064E3B; margin-top: 5px;">
                        Confidence: <strong>{conf * 100:.2f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Probability Distribution")
            st.write(f"**Parasitized (Infected):** `{prob_para * 100:.2f}%`")
            st.progress(float(prob_para))
            st.write(f"**Uninfected (Normal):** `{(1.0 - prob_para) * 100:.2f}%`")
            st.progress(float(1.0 - prob_para))

            st.markdown("### Clinical Risk Stratification")
            st.info(f"**Risk Evaluation:** {result['risk_level']}")

            with st.expander("Technical Inference Details"):
                st.json({
                    "Predicted Class": pred_class,
                    "Infection Probability P(Y=1|X)": round(prob_para, 4),
                    "Decision Threshold": threshold,
                    "Input Preprocessing": "Bilinear resize to 128x128x3, standard [0, 1] normalization",
                    "Feature Extractor": "4 Conv2D + BN + MaxPooling + GAP"
                })
        else:
            st.info("👈 Please upload or select a blood cell image on the left to view predictions.")


if __name__ == "__main__":
    main()

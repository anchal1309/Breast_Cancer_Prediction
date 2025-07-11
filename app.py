import streamlit as st
import numpy as np
import pickle

# Load model
with open('Breast_Pred.pkl', 'rb') as f:
    model = pickle.load(f)

# Set page config
st.set_page_config(page_title="🩺 Breast Cancer Prediction", layout="wide")

st.title("🔬 Breast Cancer Tumor Prediction")
st.markdown("""
This application predicts whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-cancerous)** based on various diagnostic features.
""")

# All features (excluding 'id')
features = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
    'Compactness', 'Concavity', 'Concave points', 'Symmetry', 'Fractal Dimension']
# Organize inputs into columns for better layout
input_values = []
st.subheader("📝 Input Patient Diagnostic Data")
cols = st.columns(3)

for i, feature in enumerate(features):
    col = cols[i % 3]
    value = col.text_input(f"{feature.replace('_', ' ').capitalize()}:", key=feature)
    input_values.append(value)

# Predict
if st.button("🔍 Predict Tumor Type"):
    if all(input_values):
        try:
            input_array = np.array([[int(val) for val in input_values]])
            prediction = model.predict(input_array)[0]

            # Optional: probability
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_array)[0][1]  # probability of benign

            if prediction == 0:
                st.success("🟢 **Result: Benign (Non-cancerous Tumor)**")
                if hasattr(model, "predict_proba"):
                    st.write(f"Model Confidence: **{prob:.2%}**")
            else:
                st.error("🔴 **Result: Malignant (Cancerous Tumor)**")
                if hasattr(model, "predict_proba"):
                    st.write(f"Model Confidence: **{1 - prob:.2%}**")

        except ValueError:
            st.error("❌ Please ensure all values are numeric.")
    else:
        st.warning("⚠️ Please fill in all fields to proceed.")



import streamlit as st
import pandas as pd
import joblib, json, os, numpy as np

st.set_page_config(page_title="Term Deposit Prediction", layout="centered")
st.title("💰 Term Deposit Subscription Prediction")

MODEL_FILE = "term_deposit_model.pkl"
FEATURE_FILE = "feature_columns.json"

# Load model
model = joblib.load(MODEL_FILE)
st.success("Model loaded successfully!")

# Load feature columns
with open(FEATURE_FILE, "r") as f:
    feature_cols = json.load(f)

st.write("This app expects input columns exactly matching the trained model.")

# ---------------------------
# Batch Prediction
# ---------------------------
st.header("Upload CSV for Batch Prediction")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Uploaded Data (first 5 rows)")
    st.dataframe(df.head())

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
    else:
        X = df[feature_cols]
        probs = model.predict_proba(X)[:, 1]
        df["pred_prob"] = probs
        df["pred_label"] = (probs >= 0.5).astype(int)

        st.subheader("Predictions")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode()
        st.download_button("Download Predictions", csv, file_name="predictions.csv")

# ---------------------------
# Single Prediction
# ---------------------------
st.header("Single Customer Prediction")

input_vals = {}
for col in feature_cols:
    input_vals[col] = st.text_input(col, "")

if st.button("Predict Single Record"):
    row = pd.DataFrame([input_vals])
    # Convert numeric-looking inputs
    for c in row.columns:
        try:
            row[c] = row[c].astype(float)
        except:
            pass

    prob = model.predict_proba(row)[0][1]
    label = int(prob >= 0.5)

    st.write("**Predicted Probability:**", round(prob, 3))
    st.write("**Predicted Label:**", label)

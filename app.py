import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("logreg_rfe_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎓 Admission Prediction App")

# User input
gre = st.number_input("GRE Score", min_value=260, max_value=340, step=1)
rating = st.slider("University Rating", 1, 5, value=3)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)

if st.button("Predict Admission"):
    # Prepare input
    input_data = pd.DataFrame([[1.0, gre, rating, cgpa]],
                               columns=['const', 'GRE Score', 'University Rating', 'CGPA'])
    # Predict
    prob = model.predict(input_data)[0]
    label = "Admit" if prob >= 0.6 else "Reject"
    st.success(f"Prediction: **{label}**")
    st.info(f"Probability of Admission: **{prob:.2f}**")


import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")   # or "spam_pipeline.pkl"

model = load_model()

st.title("Email Spam Detection")

st.write("Enter an email message below and the model will predict if it is Spam or Ham.")

# User input
user_text = st.text_area("Email Message")

# Predict button
if st.button("Predict"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Create dataframe matching training schema
        data = pd.DataFrame([{
            "Message": user_text,
            "Length": len(user_text)
        }])

        prediction = model.predict(data)[0]

        # Display result
        if prediction.lower() == "spam":
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: HAM")


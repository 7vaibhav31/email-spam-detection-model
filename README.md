# 📧 Email Spam Detection System

web link : https://email-spam-detection-model-7twqnotwfappsqgpsedpv4u.streamlit.app/

An end-to-end **Email Spam Detection** project built with a **clean ML pipeline**,  
hyperparameter tuning, and a **Streamlit web app** for real-time predictions.

This project follows **industry best practices** for:
- data preprocessing
- pipeline-based modeling
- hyperparameter tuning
- reproducibility
- deployment-ready inference

---

## 🚀 Project Overview

Spam emails are a common real-world problem where **accuracy alone is misleading** due to class imbalance.  
This project builds a robust spam classifier using:

- **TF-IDF** for text feature extraction
- **Message length** as an additional numeric signal
- **Logistic Regression**
- **Pipeline + ColumnTransformer** to avoid data leakage
- **RandomizedSearchCV** for hyperparameter tuning
- **Streamlit** for interactive UI

The final output is a **single saved pipeline** that can be directly deployed.

---

## 🧠 Machine Learning Approach

### Features Used
- **Email text (`Message`)** → TF-IDF Vectorization
- **Message length (`Length`)** → Standard Scaling

### Model
- Logistic Regression

### Evaluation Metric
- **F1-score / F1-macro** (to handle class imbalance)

---

## 🗂️ Project Structure

email-spam-detection/
│
├── data/
│ └── email.csv # Raw dataset
│
├── notebooks/
│ └── eda.ipynb # Exploratory Data Analysis
│
├── src/
│ ├── preprocessing.py # Data cleaning logic
│ ├── models.py # Model training & tuning
│ ├── predict.py # Inference utilities
│ └── utils.py # Helper functions
│
├── app.py # Streamlit application
├── model.pkl # Saved trained pipeline
├── config.py # Configuration & hyperparameters
├── requirements.txt
└── README.md


---

## ⚙️ How the Pipeline Works

Raw Email Text
↓
TF-IDF Vectorizer
↓
Message Length Feature
↓
Feature Combination (ColumnTransformer)
↓
Logistic Regression
↓
Spam / Ham Prediction


All preprocessing + modeling steps are encapsulated in **one pipeline**, ensuring:
- no data leakage
- consistent training & inference
- easy deployment

---

## 🔍 Hyperparameter Tuning

`RandomizedSearchCV` is used to tune:
- TF-IDF parameters (`ngram_range`, `min_df`, `max_df`, `max_features`)
- Logistic Regression parameters (`C`, `penalty`, `solver`)

Each trial:
- clones the full pipeline
- applies a new parameter set
- performs cross-validation
- selects the best performing pipeline

The **best estimator** is saved as `model.pkl`.

---

## 🧪 Example Predictions

### Spam
Congratulations! You have been selected to receive a FREE gift.
Click now to claim your reward.


### Ham
Hi,
Please find the meeting agenda attached.
Let me know if you have any questions.


---

## 🖥️ Run the Streamlit App

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Run the app
streamlit run app.py
3️⃣ Open browser
Streamlit will open automatically at:

http://localhost:8501
Enter an email message and get a real-time spam prediction.

📦 Deployment Ready
The model is saved as a single .pkl file

Can be easily deployed using:

Streamlit

FastAPI

Docker

Cloud platforms

🎯 Key Learnings
Why pipelines matter more than models

How to prevent data leakage

How to tune models correctly with text data

How to deploy ML models for real users

📌 Future Improvements
Probability-based threshold tuning

URL / punctuation based features

Model monitoring & drift detection

FastAPI backend with REST endpoints

Dockerized deployment

👤 Author
Built by Vaibhav Sharma
Focused on writing production-ready ML systems, not just notebooks.


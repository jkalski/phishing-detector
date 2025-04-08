import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model

# Load non-TF models
nb_model = joblib.load("nb_model.pkl")
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Stopwords + cleaner
stopwords = {
    "the", "and", "in", "of", "to", "a", "is", "for", "on", "that", "this",
    "with", "at", "by", "an", "be", "are", "it", "from", "or", "as", "was",
    "if", "but", "not", "have", "has", "had", "you", "your", "we", "us",
    "our", "they", "them",
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    return " ".join([word for word in words if word not in stopwords])

def explain_naive_bayes(vectorized_input, vectorizer, model, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    input_array = vectorized_input.toarray()[0]
    log_probs = model.feature_log_prob_
    diff = log_probs[1] - log_probs[0]

    influence = input_array * diff
    top_indices = np.argsort(np.abs(influence))[-top_n:][::-1]

    results = []
    for i in top_indices:
        if input_array[i] > 0:
            word = feature_names[i]
            impact = influence[i]
            results.append((word, impact))
    return results

# --- Streamlit UI ---
st.title("📧 Phishing Email Detection App")
tab1, tab2 = st.tabs(["🔍 Test One Model", "📊 Compare All Models"])

# --- TAB 1: Naive Bayes + Explanation ---
with tab1:
    st.subheader("Naive Bayes Email Detector")
    email_input = st.text_area("Paste email content:", height=200)

    if st.button("Check Email", key="single"):
        cleaned = clean_text(email_input)
        vectorized = vectorizer.transform([cleaned])

        prediction = nb_model.predict(vectorized)[0]
        confidence = nb_model.predict_proba(vectorized)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 Likely phishing (Confidence: {confidence:.2%})")
        else:
            st.success(f"✅ Likely legitimate (Confidence: {confidence:.2%})")

        st.markdown("#### 🧠 Top Words Influencing Prediction")
        top_words = explain_naive_bayes(vectorized, vectorizer, nb_model)
        explanation_df = []

        for word, impact in top_words:
            label = "Phishing" if impact > 0 else "Legit"
            emoji = "🟥" if impact > 0 else "🟩"
            score = f"{impact:.2f}"
            explanation_df.append([word, score, f"{emoji} {label}"])

        st.table(pd.DataFrame(explanation_df, columns=["Word", "Impact Score", "Leans Toward"]))

# --- TAB 2: Compare All Models ---
with tab2:
    st.subheader("Compare All Models Side by Side")
    compare_input = st.text_area("Paste email content to compare:", height=200)

    if st.button("Run Comparison", key="compare"):
        cleaned = clean_text(compare_input)
        vectorized = vectorizer.transform([cleaned])
        dense_input = vectorized.toarray()

        # Naive Bayes
        nb_pred = nb_model.predict(vectorized)[0]
        nb_conf = nb_model.predict_proba(vectorized)[0][nb_pred]

        # SVM
        svm_pred = svm_model.predict(vectorized)[0]
        svm_conf = 1.0  # No probabilities

        # Neural Net (loaded safely at runtime)
        nn_model = load_model("nn_model.keras")
        with tf.device("/CPU:0"):
            nn_out = nn_model.predict(dense_input, verbose=0)[0]
        nn_pred = np.argmax(nn_out)
        nn_conf = nn_out[nn_pred]

        st.markdown("### 📈 Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Naive Bayes")
            st.write(f"Prediction: {'Phishing' if nb_pred == 1 else 'Legit'}")
            st.write(f"Confidence: {nb_conf:.2%}")

        with col2:
            st.subheader("SVM")
            st.write(f"Prediction: {'Phishing' if svm_pred == 1 else 'Legit'}")
            st.write(f"Confidence: {svm_conf:.2%}*")

        with col3:
            st.subheader("Neural Net")
            st.write(f"Prediction: {'Phishing' if nn_pred == 1 else 'Legit'}")
            st.write(f"Confidence: {nn_conf:.2%}")

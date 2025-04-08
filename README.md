# 📧 Phishing Email Detection App

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://phishing-detector-feytfxdzppy4tnz3aynpd9.streamlit.app)

A real-time phishing email detection web app built with **Streamlit** and powered by **machine learning models** trained on 80,000+ emails. Users can paste email content and instantly receive predictions from multiple models on whether the email is **phishing** or **legitimate**.



### 📺 Live Demo

👉 Try it here:  
[https://phishing-detector-feytfxdzppy4tnz3aynpd9.streamlit.app](https://phishing-detector-feytfxdzppy4tnz3aynpd9.streamlit.app)

Paste in a suspicious-looking email and compare predictions from Naive Bayes, SVM, and Neural Network models.



## 🚀 Features

- 🔍 Predicts phishing vs. legit emails
- 📈 Confidence scores from:
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Neural Network
- 📊 Compare models side-by-side
- 🧠 Displays key influencing words (NB explanation tab)
- 🖥 Streamlit-powered UI



## 🧠 Models & Dataset

- Dataset size: **~82,500 emails**
  - 42,891 phishing / spam
  - 39,595 legitimate
- Sources: Enron, SpamAssassin, Nigerian Fraud, CEAS, etc.
- Models trained using:
  - TF-IDF vectorization
  - Scikit-learn & TensorFlow/Keras
- Note: The full dataset was removed from the repo due to GitHub’s size limits



## 🛠 Tech Stack

- `Python`
- `scikit-learn`
- `TensorFlow / Keras`
- `pandas`, `joblib`
- `Streamlit`

![image](https://github.com/user-attachments/assets/c657486b-d98b-4bbb-b56e-4137ac2474bc)

Justin Kalski
CS Student @ CSU Northridge

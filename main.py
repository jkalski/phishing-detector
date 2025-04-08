import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib  # for saving model and vectorizer

# Load the dataset
df = pd.read_csv("phishing_email.csv")

# Basic stopwords
stopwords = {
    'the', 'and', 'in', 'of', 'to', 'a', 'is', 'for', 'on', 'that', 'this',
    'with', 'at', 'by', 'an', 'be', 'are', 'it', 'from', 'or', 'as', 'was',
    'if', 'but', 'not', 'have', 'has', 'had', 'you', 'your', 'we', 'us',
    'our', 'they', 'them'
}

# Simple text cleaner
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return ' '.join([word for word in words if word not in stopwords])

# Clean all emails
df['cleaned_text'] = df['text_combined'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved!")

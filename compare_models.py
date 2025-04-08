import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load and clean data
df = pd.read_csv("phishing_email.csv")

stopwords = {
    'the', 'and', 'in', 'of', 'to', 'a', 'is', 'for', 'on', 'that', 'this',
    'with', 'at', 'by', 'an', 'be', 'are', 'it', 'from', 'or', 'as', 'was',
    'if', 'but', 'not', 'have', 'has', 'had', 'you', 'your', 'we', 'us',
    'our', 'they', 'them'
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return ' '.join([word for word in words if word not in stopwords])

df['cleaned_text'] = df['text_combined'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

# Save vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
print("📦 Naive Bayes Results:\n", classification_report(y_test, nb_pred))
joblib.dump(nb_model, "nb_model.pkl")

# 2. SVM
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print("⚙️ SVM Results:\n", classification_report(y_test, svm_pred))
joblib.dump(svm_model, "svm_model.pkl")

# 3. Neural Network (Keras)
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

nn_model = Sequential([
    Input(shape=(5000,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3)

nn_model.fit(
    X_train_dense,
    y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

nn_eval = nn_model.evaluate(X_test_dense, y_test_cat)
print(f"🧠 Neural Net Accuracy: {nn_eval[1]:.2%}")
nn_model.save("nn_model.keras")

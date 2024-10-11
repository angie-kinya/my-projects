import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset 
data = pd.read_csv('movie/data/movie.csv')

print(data.head())

# Preprocessing function to clean the text
def preprocess_text(text):
    #Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

# Apply preprocessing to the 'review' column
data['clean_text'] = data['text'].apply(preprocess_text)

# Split the data into train and test sets
X = data['clean_text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data size: {len(X_train)}")
print(f"Test data size: {len(X_test)}")


# Initialize TF-IDF Vectorizer (Term Frequency-Inverse Document Frequency)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

# Transform the training and test data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Shape of TF-IDF matrix for training data: {X_train_tfidf.shape}")


# Initialize logistic regression model
model = LogisticRegression()

# Train the model on the TF-IDF matrix of training data
model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_tfidf)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report (Precision - exactness, Recall - completeness, F1-score - balance between precision and recall)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# Test with a new review
new_text = ["The movie was absolutely fantastic and thrilling!"]
new_text_clean = preprocess_text(new_text[0])
new_text_tfidf = tfidf.transform([new_text_clean])

# Predict label
predicted_label = model.predict(new_text_tfidf)
print(f"Predicted Sentiment: {predicted_label[0]}")
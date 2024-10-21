import pandas as pd
import numpy as np
import re
import nltk
import gensim.downloader as api
import shap
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('movie/data/amazon_reviews.csv')
print(data.head())

# Preprocess the anamzon reviews
def preprocess_text(text):
    if isinstance(text, str):  # Check if the input is a string
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower().strip()
        return text
    else:
        return ''  # Return an empty string for missing or null values

# Apply preprocessing
data['clean_review'] = data['reviewText'].apply(preprocess_text)
data['sentiment'] = data['overall'].apply(lambda x: 'positive' if x > 3 else 'negative')

X = data['clean_review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', use_idf=False)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
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

# ADVANCED TEXT PROCESSING
# Lemmatization
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

data['clean_review'] = data['clean_review'].apply(lemmatize_text)


# DIFFERENT VECTORIZATION METHOD
# Load pre-trained GloVe embeddings
glove_vectors = api.load("glove-wiki-gigaword-100")

# Convert reviews to embeddings (average of word embeddings)
def embed_review(review):
    words = review.split()
    word_vecs = [glove_vectors[word] for word in words if word in glove_vectors]
    if len(word_vecs) == 0:
        return np.zeros(100) # 100 is the size of the GloVe vector
    return np.mean(word_vecs, axis=0)

# Apply embedding to the entire dataset
X_train_embedded = np.array([embed_review(review) for review in X_train])
X_test_embedded = np.array([embed_review(review) for review in X_test])

# Standardize the data
scaler = StandardScaler()
X_train_embedded = scaler.fit_transform(X_train_embedded)
X_test_embedded = scaler.transform(X_test_embedded)

# NAIVE BAYES MODEL
# Initialize Naive Bayes Model
nb_model = MultinomialNB()

# Train the model
nb_model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred_nb = nb_model.predict(X_test_tfidf)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb * 100:.2f}%")

# HYPERPARAMETER TUNING
# Define a parameter grid for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10], # Regularization strength
    'solver':  ['newton-cg', 'lbfgs', 'liblinear'] #  Optimization algorithms
}

#  Perform grid search with cross-validation
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

# Best parameters and accuracy
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_}")

# MODEL EXPLAINABILITY  WITH SHAP (SHapley Additive exPlanations)
# Initialize the SHAP explainer with the trained model
explainer = shap.LinearExplainer(model, X_train_tfidf)

# Get SHAP values for the test set
shap_values = explainer.shap_values(X_test_tfidf)

# Get the feature names from the TF-IDF vectorizer
feature_names = tfidf.get_feature_names_out()

# Convert the first review back to a format suitable for SHAP
first_review_vector = X_test_tfidf[0].toarray()  # Convert sparse matrix to dense array

# Plot the SHAP values for the first test review
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], feature_names, first_review_vector)

# DEPLOY THE MODEL
# Title
st.title("Sentiment Analysis App")

# Input review from the user
user_review = st.text_area("Enter a movie review:")

# Predict the sentiment
if st.button("Predict"):
    user_review_clean = preprocess_text(user_review)
    user_review_tfidf = tfidf.transform([user_review_clean])
    predicted_sentiment = model.predict(user_review_tfidf)
    st.write(f"Predicted Sentiment: {predicted_sentiment[0]}")
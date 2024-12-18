import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
import matplotlib.pyplot as plt
import shap
from lime.lime_text import LimeTextExplainer
import numpy as np

# Parameters
VOCAB_SIZE = 10000  # Number of most frequent words to keep
TEST_SUBSET_SIZE = 1000  # Number of samples from the test set for permutation importance

# Step 1: Load the IMDB Dataset from CSV
def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Dataset Loaded. Sample:")
    print(data.head())
    return data

# Step 2: Preprocess Data (Text Vectorization with TF-IDF)
def preprocess_data(data, vocab_size=VOCAB_SIZE):
    vectorizer = TfidfVectorizer(max_features=vocab_size, stop_words='english')
    X_tfidf = vectorizer.fit_transform(data['review'])
    return X_tfidf, vectorizer

# Step 3: Train Logistic Regression Model
def train_logistic_regression(X_train, y_train):
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

# Step 4: Evaluate Model Accuracy
def evaluate_model(lr_model, X_test, y_test):
    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Model Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Step 5: Compute Permutation Importance
def compute_permutation_importance(lr_model, X_test, y_test, vectorizer, test_subset_size=TEST_SUBSET_SIZE):
    # Use a subset of the test set to reduce memory usage
    X_test_subset, y_test_subset = resample(X_test, y_test, n_samples=test_subset_size, random_state=42)
    X_test_subset_dense = X_test_subset.toarray()  # Convert to dense format for permutation importance

    # Compute permutation importance
    result = permutation_importance(
        lr_model, X_test_subset_dense, y_test_subset, n_repeats=10, n_jobs=-1, random_state=42
    )
    permutation_importances = result.importances_mean

    # Map permutation importance to feature names
    feature_names = vectorizer.get_feature_names_out()
    feature_importance_perm = sorted(
        zip(feature_names, permutation_importances), key=lambda x: x[1], reverse=True
    )

    # Display the top 20 features by permutation importance
    print("\nTop 20 features by permutation importance:")
    for feature, importance in feature_importance_perm[:20]:
        print(f"Feature: {feature}, Permutation Importance: {importance:.4f}")

# Main function to run all steps
def main(file_path):
    # Load the data
    data = load_data(file_path)
    
    # Preprocess the data (vectorize reviews with TF-IDF)
    X_tfidf, vectorizer = preprocess_data(data)

    # Map sentiments to binary labels (positive -> 1, negative -> 0)
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['sentiment'].values, test_size=0.3, random_state=42)

    # Train the logistic regression model
    lr_model = train_logistic_regression(X_train, y_train)

    # Evaluate the model
    evaluate_model(lr_model, X_test, y_test)

    # Compute permutation importance of the features
    compute_permutation_importance(lr_model, X_test, y_test, vectorizer)

if __name__ == "__main__":
    file_path = "IMDB Dataset.csv/IMDB Dataset.csv"  # Replace with your actual CSV file path
    main(file_path)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import shap
from lime.lime_text import LimeTextExplainer
import numpy as np
import os

# Prevent matplotlib from blocking code execution
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid blocking

# Parameters
VOCAB_SIZE = 10000  # Number of most frequent words to keep
TEST_SUBSET_SIZE = 1000  # Subset size for permutation importance

# Step 1: Load and prepare the IMDB Dataset
data = pd.read_csv("IMDB Dataset.csv/IMDB Dataset.csv")  # Load dataset

# Ensure sentiment column is binary
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Step 2: Preprocess Data with TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE, stop_words='english')
X_tfidf = vectorizer.fit_transform(data['review'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_tfidf, data['sentiment'], test_size=0.3, random_state=42)

# Step 3: Train Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

# Evaluate Logistic Regression Model
y_pred = lr_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Test Accuracy: {accuracy * 100:.2f}%\n")

# Step 4: Partial Dependence Plot (Feature Contribution Visualization)
# Downsample training data for efficiency
x_sample, y_sample = resample(x_train, y_train, n_samples=1000, random_state=42)
x_sample_dense = x_sample.toarray()

print("Generating Partial Dependence Plot...")
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(lr_model, x_sample_dense, [0, 1], ax=ax)
plt.title("Partial Dependence Plot")
plt.savefig('partial_dependence_plot.png')
plt.close()
print("Partial Dependence Plot saved as 'partial_dependence_plot.png'.\n")

# Step 5: SHAP (SHapley Additive Explanations) for Global Feature Importance
print("Generating SHAP summary plot...")
explainer = shap.Explainer(lr_model, x_train)
shap_values = explainer(x_test)
shap.summary_plot(shap_values, feature_names=vectorizer.get_feature_names_out(), plot_type="bar", show=False)
plt.savefig('shap_summary_plot.png')
plt.close()
print("SHAP summary plot saved as 'shap_summary_plot.png'.\n")

# Step 6: LIME (Local Interpretable Model-Agnostic Explanations)
pipeline = make_pipeline(vectorizer, lr_model)
explainer = LimeTextExplainer(class_names=['negative', 'positive'])

# Explain an instance from the test set
instance_idx = 26  # Example index
instance_text = data['review'].iloc[x_test[instance_idx].indices[0]]
explanation = explainer.explain_instance(instance_text, pipeline.predict_proba, num_features=10)

print(f"\nInstance to Explain:\n{instance_text}\n")
print("LIME Explanation (Top Features):")
for feature, weight in explanation.as_list():
    print(f"{feature}: {weight:.4f}")

explanation.save_to_file('lime_explanation.html')
print("LIME explanation saved to lime_explanation.html\n")

# Step 7: Feature Importance and Selection
# Tree-based feature importance for Logistic Regression
print("Top Features Based on Logistic Regression Coefficients:")
feature_importances = np.abs(lr_model.coef_[0])
sorted_idx = np.argsort(feature_importances)[::-1]
for idx in sorted_idx[:10]:
    print(f"{vectorizer.get_feature_names_out()[idx]}: {feature_importances[idx]:.4f}")

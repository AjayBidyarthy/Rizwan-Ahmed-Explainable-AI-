# README

## Overview
This repository contains scripts for implementing various explainability techniques on different machine learning models and datasets. These codes aim to enhance our understanding of the importance of transparency in algorithmic decision-making.

### File Descriptions

#### `CNN_Model_and_Techniques_1.py`
This script demonstrates the application of **Integrated Gradients** and **SmoothGrad** techniques for explainability. The code is implemented using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.

#### `CNN_Model_and_Techniques_2.py`
This script showcases the use of the **Grad-CAM** technique for visualizing model predictions. The model used is a **CNN** trained on the **MNIST dataset**.

#### `LogisticRegression_techniques.py`
This script applies various explainability techniques, including:
- **SHAP (SHapley Additive exPlanations)**
- **LIME (Local Interpretable Model-agnostic Explanations)**
- **Partial Dependence Plots (PDP)**
- **Tree-based Feature Importance**

The techniques are applied to a **Logistic Regression** model trained on the **IMDB review dataset** for sentiment analysis. The dataset can be accessed from [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

#### `permutation_importance.py`
This script explains the **Permutation Importance** technique, applied to a **Logistic Regression** model. The model is trained on the **IMDB review dataset** to demonstrate sentiment analysis. The dataset can be accessed from [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Usage
1. Clone the repository to your local machine.
2. Install the necessary dependencies listed in the `requirements.txt` file.
3. Run the desired script to explore the respective explainability technique.

## Dependencies
- Python (version >= 3.11)
- TensorFlow/Keras (for CNN-related scripts)
- Scikit-learn
- SHAP
- LIME
- Matplotlib

## Datasets
- **MNIST dataset**: Used for CNN-based models and techniques.
- **IMDB review dataset**: Used for Logistic Regression models and techniques. Access it [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).


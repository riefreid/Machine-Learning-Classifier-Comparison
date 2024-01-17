# Machine Learning Classifier Comparison

This repository contains Python code for comparing the performance of various machine learning classifiers on a dataset. The classifiers include Decision Tree, K-Nearest Neighbors (KNN), Logistic Regression, Multi-Layer Perceptron (MLP), Random Forest, Support Vector Machine (SVM), and XGBoost. The dataset is loaded from 'dataset.csv' and undergoes preprocessing steps such as data cleaning, normalization, and encoding of class labels.

## Classifier Comparison

The code compares the accuracy of each classifier under different normalization techniques: No Normalization, Z-Score Normalization, and Min-Max Normalization. It also explores hyperparameter tuning for the SVM classifier using GridSearchCV.

## Usage

1. Clone the repository:
   git clone https://github.com/your-username/Machine-Learning-Classifier-Comparison.git
2. Run the Python script:
   python classifier_comparison.py
3. Explore the classifier comparison results and hyperparameter tuning.

## Dependencies
- pandas
- scikit-learn
- xgboost
- matplotlib
- keras (from tensorflow)

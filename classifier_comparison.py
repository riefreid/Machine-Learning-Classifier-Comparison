import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow import keras  # Import the entire keras module from tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Read the CSV file
data = pd.read_csv('dataset.csv')

# Store the class labels
class_labels = data['label']

# Drop 'Barcode' and 'label' columns and the first row (column headers)
data = data.drop(columns=['Barcode', 'label'])
data = data.iloc[1:]

# Convert the data to numeric (if needed)
data = data.apply(pd.to_numeric, errors='coerce')

# Ensure shapes are aligned
class_labels = class_labels.iloc[1:]

# Create a variable with no normalization
data_no_norm = data.copy()

# Create a variable with z-score normalization
scaler = StandardScaler()
data_zscore_norm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Create a variable with min-max normalization
min_max_scaler = MinMaxScaler()
data_minmax_norm = pd.DataFrame(min_max_scaler.fit_transform(data), columns=data.columns)

# Split your dataset and divide into the training set and test set; fix random_state for reproducibility
x_train_no_norm, x_test_no_norm, y_train, y_test = train_test_split(data_no_norm, class_labels, test_size=0.25, random_state=0)
x_train_zscore_norm, x_test_zscore_norm, y_train, y_test = train_test_split(data_zscore_norm, class_labels, test_size=0.25, random_state=0)
x_train_minmax_norm, x_test_minmax_norm, y_train, y_test = train_test_split(data_minmax_norm, class_labels, test_size=0.25, random_state=0)  from sklearn.preprocessing import LabelEncoder


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Encode the class labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Decision Tree / Random Forest / XGBoost
# Decision Tree Classifier
dt_clf_no_norm = DecisionTreeClassifier()
dt_clf_zscore_norm = DecisionTreeClassifier()
dt_clf_minmax_norm = DecisionTreeClassifier()

# Random Forest Classifier
rf_clf_no_norm = RandomForestClassifier()
rf_clf_zscore_norm = RandomForestClassifier()
rf_clf_minmax_norm = RandomForestClassifier()

# XGBoost Classifier
xgb_clf_no_norm = xgb.XGBClassifier()
xgb_clf_zscore_norm = xgb.XGBClassifier()
xgb_clf_minmax_norm = xgb.XGBClassifier()

# Fit models using training data
dt_clf_no_norm.fit(x_train_no_norm, y_train_encoded)
rf_clf_no_norm.fit(x_train_no_norm, y_train_encoded)
xgb_clf_no_norm.fit(x_train_no_norm, y_train_encoded)

dt_clf_zscore_norm.fit(x_train_zscore_norm, y_train_encoded)
rf_clf_zscore_norm.fit(x_train_zscore_norm, y_train_encoded)
xgb_clf_zscore_norm.fit(x_train_zscore_norm, y_train_encoded)

dt_clf_minmax_norm.fit(x_train_minmax_norm, y_train_encoded)
rf_clf_minmax_norm.fit(x_train_minmax_norm, y_train_encoded)
xgb_clf_minmax_norm.fit(x_train_minmax_norm, y_train_encoded)

# Predict the test data
pred_dt_no_norm = dt_clf_no_norm.predict(x_test_no_norm)
pred_rf_no_norm = rf_clf_no_norm.predict(x_test_no_norm)
pred_xgb_no_norm = xgb_clf_no_norm.predict(x_test_no_norm)

pred_dt_zscore_norm = dt_clf_zscore_norm.predict(x_test_zscore_norm)
pred_rf_zscore_norm = rf_clf_zscore_norm.predict(x_test_zscore_norm)
pred_xgb_zscore_norm = xgb_clf_zscore_norm.predict(x_test_zscore_norm)

pred_dt_minmax_norm = dt_clf_minmax_norm.predict(x_test_minmax_norm)
pred_rf_minmax_norm = rf_clf_minmax_norm.predict(x_test_minmax_norm)
pred_xgb_minmax_norm = xgb_clf_minmax_norm.predict(x_test_minmax_norm)

# Calculate accuracies
accuracy_dt_no_norm = accuracy_score(y_test_encoded, pred_dt_no_norm)
accuracy_rf_no_norm = accuracy_score(y_test_encoded, pred_rf_no_norm)
accuracy_xgb_no_norm = accuracy_score(y_test_encoded, pred_xgb_no_norm)

accuracy_dt_zscore_norm = accuracy_score(y_test_encoded, pred_dt_zscore_norm)
accuracy_rf_zscore_norm = accuracy_score(y_test_encoded, pred_rf_zscore_norm)
accuracy_xgb_zscore_norm = accuracy_score(y_test_encoded, pred_xgb_zscore_norm)

accuracy_dt_minmax_norm = accuracy_score(y_test_encoded, pred_dt_minmax_norm)
accuracy_rf_minmax_norm = accuracy_score(y_test_encoded, pred_rf_minmax_norm)
accuracy_xgb_minmax_norm = accuracy_score(y_test_encoded, pred_xgb_minmax_norm)

# Print accuracies
print("Decision Tree Accuracy (No Normalization):", accuracy_dt_no_norm)
print("Decision Tree Accuracy (Z-Score Normalization):", accuracy_dt_zscore_norm)
print("Decision Tree Accuracy (Min-Max Normalization):", accuracy_dt_minmax_norm)

print("\nRandom Forest Accuracy (No Normalization):", accuracy_rf_no_norm)
print("Random Forest Accuracy (Z-Score Normalization):", accuracy_rf_zscore_norm)
print("Random Forest Accuracy (Min-Max Normalization):", accuracy_rf_minmax_norm)

print("\nXGBoost Accuracy (No Normalization):", accuracy_xgb_no_norm)
print("XGBoost Accuracy (Z-Score Normalization):", accuracy_xgb_zscore_norm)
print("XGBoost Accuracy (Min-Max Normalization):", accuracy_xgb_minmax_norm)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# K-Nearest Neighbors
# KNN Classifier
knn_clf_no_norm = KNeighborsClassifier()
knn_clf_zscore_norm = KNeighborsClassifier()
knn_clf_minmax_norm = KNeighborsClassifier()

# Encode the class labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Fit models using training data
knn_clf_no_norm.fit(x_train_no_norm, y_train_encoded)
knn_clf_zscore_norm.fit(x_train_zscore_norm, y_train_encoded)
knn_clf_minmax_norm.fit(x_train_minmax_norm, y_train_encoded)

# Predict the test data
pred_knn_no_norm = knn_clf_no_norm.predict(x_test_no_norm)
pred_knn_zscore_norm = knn_clf_zscore_norm.predict(x_test_zscore_norm)
pred_knn_minmax_norm = knn_clf_minmax_norm.predict(x_test_minmax_norm)

# Calculate accuracies
accuracy_knn_no_norm = accuracy_score(y_test_encoded, pred_knn_no_norm)
accuracy_knn_zscore_norm = accuracy_score(y_test_encoded, pred_knn_zscore_norm)
accuracy_knn_minmax_norm = accuracy_score(y_test_encoded, pred_knn_minmax_norm)

# Print accuracies
print("KNN Accuracy (No Normalization):", accuracy_knn_no_norm)
print("KNN Accuracy (Z-Score Normalization):", accuracy_knn_zscore_norm)
print("KNN Accuracy (Min-Max Normalization):", accuracy_knn_minmax_norm)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Logistic Regression
logisticRegr_no_norm = LogisticRegression(max_iter=1200)
logisticRegr_zscore_norm = LogisticRegression(max_iter=1200)
logisticRegr_minmax_norm = LogisticRegression(max_iter=1200)

# Encode the class labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Fit models using training data
logisticRegr_no_norm.fit(x_train_no_norm, y_train_encoded)
logisticRegr_zscore_norm.fit(x_train_zscore_norm, y_train_encoded)
logisticRegr_minmax_norm.fit(x_train_minmax_norm, y_train_encoded)

# Predict the test data
pred_no_norm = logisticRegr_no_norm.predict(x_test_no_norm)
pred_zscore_norm = logisticRegr_zscore_norm.predict(x_test_zscore_norm)
pred_minmax_norm = logisticRegr_minmax_norm.predict(x_test_minmax_norm)

# Calculate accuracies
accuracy_no_norm = accuracy_score(y_test_encoded, pred_no_norm)
accuracy_zscore_norm = accuracy_score(y_test_encoded, pred_zscore_norm)
accuracy_minmax_norm = accuracy_score(y_test_encoded, pred_minmax_norm)

# Print accuracies
print("Logistic Regression Accuracy (No Normalization):", accuracy_no_norm)
print("Logistic Regression Accuracy (Z-Score Normalization):", accuracy_zscore_norm)
print("Logistic Regression Accuracy (Min-Max Normalization):", accuracy_minmax_norm)


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Encode the class labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Multi-Layer Perceptron
def mlp_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(250, input_dim=input_dim, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Create MLP models
mlp_clf_no_norm = mlp_model(x_train_no_norm.shape[1], len(label_encoder.classes_))
mlp_clf_zscore_norm = mlp_model(x_train_zscore_norm.shape[1], len(label_encoder.classes_))
mlp_clf_minmax_norm = mlp_model(x_train_minmax_norm.shape[1], len(label_encoder.classes_))

# Compile models
mlp_clf_no_norm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
mlp_clf_zscore_norm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
mlp_clf_minmax_norm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit models
mlp_clf_no_norm.fit(x_train_no_norm, y_train_encoded, epochs=100, batch_size=128)
mlp_clf_zscore_norm.fit(x_train_zscore_norm, y_train_encoded, epochs=100, batch_size=128)
mlp_clf_minmax_norm.fit(x_train_minmax_norm, y_train_encoded, epochs=100, batch_size=128)

# Predict the test data
pred_no_norm_probs = mlp_clf_no_norm.predict(x_test_no_norm)
pred_zscore_norm_probs = mlp_clf_zscore_norm.predict(x_test_zscore_norm)
pred_minmax_norm_probs = mlp_clf_minmax_norm.predict(x_test_minmax_norm)

# Get predicted classes
pred_no_norm = np.argmax(pred_no_norm_probs, axis=1)
pred_zscore_norm = np.argmax(pred_zscore_norm_probs, axis=1)
pred_minmax_norm = np.argmax(pred_minmax_norm_probs, axis=1)

# Calculate accuracies
accuracy_no_norm = accuracy_score(y_test_encoded, pred_no_norm)
accuracy_zscore_norm = accuracy_score(y_test_encoded, pred_zscore_norm)
accuracy_minmax_norm = accuracy_score(y_test_encoded, pred_minmax_norm)

# Print accuracies
print("MLP Accuracy (No Normalization):", accuracy_no_norm)
print("MLP Accuracy (Z-Score Normalization):", accuracy_zscore_norm)
print("MLP Accuracy (Min-Max Normalization):", accuracy_minmax_norm)from sklearn.preprocessing import LabelEncoder


from sklearn.svm import SVC

# Encode the class labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Support Vector Machine (SVM)
svm_clf_no_norm = SVC()
svm_clf_zscore_norm = SVC()
svm_clf_minmax_norm = SVC()

# Fit models using training data
svm_clf_no_norm.fit(x_train_no_norm, y_train_encoded)
svm_clf_zscore_norm.fit(x_train_zscore_norm, y_train_encoded)
svm_clf_minmax_norm.fit(x_train_minmax_norm, y_train_encoded)

# Predict the test data
pred_no_norm = svm_clf_no_norm.predict(x_test_no_norm)
pred_zscore_norm = svm_clf_zscore_norm.predict(x_test_zscore_norm)
pred_minmax_norm = svm_clf_minmax_norm.predict(x_test_minmax_norm)

# Calculate accuracies
accuracy_no_norm = accuracy_score(y_test_encoded, pred_no_norm)
accuracy_zscore_norm = accuracy_score(y_test_encoded, pred_zscore_norm)
accuracy_minmax_norm = accuracy_score(y_test_encoded, pred_minmax_norm)

# Print accuracies
print("SVM Accuracy (No Normalization):", accuracy_no_norm)
print("SVM Accuracy (Z-Score Normalization):", accuracy_zscore_norm)
print("SVM Accuracy (Min-Max Normalization):", accuracy_minmax_norm)import matplotlib.pyplot as plt

# List of classifiers
classifiers = ['Decision Tree', 'KNN', 'Logistic Regression', 'MLP', 'Random Forest', 'SVM', 'XGBoost']

# List of accuracy values
accuracies_no_norm = [accuracy_dt_no_norm, accuracy_knn_no_norm, accuracy_no_norm, accuracy_no_norm, accuracy_rf_no_norm, accuracy_no_norm, accuracy_xgb_no_norm]
accuracies_zscore_norm = [accuracy_dt_zscore_norm, accuracy_knn_zscore_norm, accuracy_zscore_norm, accuracy_zscore_norm, accuracy_rf_zscore_norm, accuracy_zscore_norm, accuracy_xgb_zscore_norm]
accuracies_minmax_norm = [accuracy_dt_minmax_norm, accuracy_knn_minmax_norm, accuracy_minmax_norm, accuracy_minmax_norm, accuracy_rf_minmax_norm, accuracy_minmax_norm, accuracy_xgb_minmax_norm]

# Bar width
bar_width = 0.25

# Set up positions for the bars
r1 = range(len(classifiers))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plotting the bars
fig, ax = plt.subplots(figsize=(12, 8))  # Larger graph
bar1 = ax.bar(r1, accuracies_no_norm, color='purple', width=bar_width, edgecolor='grey', label='No Normalization')
bar2 = ax.bar(r2, accuracies_zscore_norm, color='navy', width=bar_width, edgecolor='grey', label='Z-Score Normalization')
bar3 = ax.bar(r3, accuracies_minmax_norm, color='red', width=bar_width, edgecolor='grey', label='Min-Max Normalization')

# Add labels, title, and legend
ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy')
ax.set_title('Classifier Accuracies by Normalization')
ax.set_xticks([r + bar_width for r in range(len(classifiers))])
ax.set_xticklabels(classifiers)

# Display accuracy values on top of the bars at a 45-degree angle
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}', ha='center', va='bottom', rotation=45)

autolabel(bar1)
autolabel(bar2)
autolabel(bar3)

# Move the legend to the upper left corner
ax.legend(loc='lower left')

# Show the plot
plt.show()


import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Read the CSV file and preprocess data
data = pd.read_csv('dataset.csv')

# Store the class labels
class_labels = data['label']

# Drop 'Barcode' and 'label' columns and the first row (column headers)
data = data.drop(columns=['Barcode', 'label'])
data = data.iloc[1:]

# Convert the data to numeric (if needed)
data = data.apply(pd.to_numeric, errors='coerce')

# Ensure shapes are aligned
class_labels = class_labels.iloc[1:]

# Define the hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.1, 1, 10, 100]
}

# Create the SVM classifier
svm_clf = SVC()

# Perform GridSearchCV with 5-fold CV
grid_search = GridSearchCV(estimator=svm_clf, param_grid=param_grid, cv=5, scoring='accuracy')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, class_labels, test_size=0.2)

# Fit the GridSearchCV model
grid_search.fit(X_train, y_train)

# Get the best model and its score
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Print the best parameters and test accuracy
print("Best parameters:", grid_search.best_params_)
print("Test accuracy:", test_accuracy)


from sklearn.preprocessing import LabelEncoder

# Encode the class labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Support Vector Machine (SVM)
svm_clf_no_norm = SVC()
svm_clf_zscore_norm = SVC()
svm_clf_minmax_norm = SVC()
# SVM - Optimal Values
optimal_svm_clf_no_norm = SVC(C=0.1, kernel='linear', gamma=0.1)
optimal_svm_clf_zscore_norm = SVC(C=0.1, kernel='linear', gamma=0.1)
optimal_svm_clf_minmax_norm = SVC(C=0.1, kernel='linear', gamma=0.1)

# Fit models using training data
svm_clf_no_norm.fit(x_train_no_norm, y_train_encoded)
svm_clf_zscore_norm.fit(x_train_zscore_norm, y_train_encoded)
svm_clf_minmax_norm.fit(x_train_minmax_norm, y_train_encoded)
optimal_svm_clf_no_norm.fit(x_train_no_norm, y_train_encoded)
optimal_svm_clf_zscore_norm.fit(x_train_zscore_norm, y_train_encoded)
optimal_svm_clf_minmax_norm.fit(x_train_minmax_norm, y_train_encoded)

# Predict the test data
pred_no_norm = svm_clf_no_norm.predict(x_test_no_norm)
pred_zscore_norm = svm_clf_zscore_norm.predict(x_test_zscore_norm)
pred_minmax_norm = svm_clf_minmax_norm.predict(x_test_minmax_norm)
pred_op_no_norm = optimal_svm_clf_no_norm.predict(x_test_no_norm)
pred_op_zscore_norm = optimal_svm_clf_zscore_norm.predict(x_test_zscore_norm)
pred_op_minmax_norm = optimal_svm_clf_minmax_norm.predict(x_test_minmax_norm)

# Calculate accuracies
accuracy_no_norm = accuracy_score(y_test_encoded, pred_no_norm)
accuracy_zscore_norm = accuracy_score(y_test_encoded, pred_zscore_norm)
accuracy_minmax_norm = accuracy_score(y_test_encoded, pred_minmax_norm)
accuracy_op_no_norm = accuracy_score(y_test_encoded, pred_op_no_norm)
accuracy_op_zscore_norm = accuracy_score(y_test_encoded, pred_op_zscore_norm)
accuracy_op_minmax_norm = accuracy_score(y_test_encoded, pred_op_minmax_norm)

# Print accuracies
print("SVM Accuracy (No Normalization):", accuracy_no_norm)
print("Optimized SVM Accuracy (No Normalization):", accuracy_op_no_norm)

print("SVM Accuracy (Z-Score Normalization):", accuracy_zscore_norm)
print("Optimized SVM Accuracy (Z-Score Normalization):", accuracy_op_zscore_norm)

print("SVM Accuracy (Min-Max Normalization):", accuracy_minmax_norm)
print("Optimized SVM Accuracy (Min-Max Normalization):", accuracy_op_minmax_norm)import matplotlib.pyplot as plt


import numpy as np

# List of normalization techniques
normalization_techniques = ['None', 'Z-Score', 'Min-Max']

# List of accuracies
accuracies = [
    accuracy_no_norm, accuracy_zscore_norm, accuracy_minmax_norm,
    accuracy_op_no_norm, accuracy_op_zscore_norm, accuracy_op_minmax_norm
]

# Reshape the accuracies for plotting
accuracies_reshaped = [accuracies[i:i + 3] for i in range(0, len(accuracies), 3)]

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))

# Plotting bars for each normalization technique
bar_width = 0.25  # Changed bar width
for i, acc in enumerate(accuracies_reshaped):
    x_positions = np.arange(len(normalization_techniques)) + i * bar_width
    ax.bar(x_positions, acc, width=bar_width, label=f'Model {i + 1}')

# Adding labels and title
ax.set_xlabel('Normalization Type')
ax.set_ylabel('Accuracy')
ax.set_title('SVM Accuracies Grouped by Normalization Type')
ax.legend(['Original', 'Optimized'])

# Set x-axis ticks and labels, centering them
ax.set_xticks(np.arange(len(normalization_techniques)) + bar_width)
ax.set_xticklabels(normalization_techniques)

# Display accuracy values at a 45-degree angle
for i, acc in enumerate(accuracies_reshaped):
    for j, value in enumerate(acc):
        ax.text(j + i * bar_width + bar_width / 2, value, f'{value:.3f}', rotation=45, ha='center', va='bottom')

# Show the plot
plt.show()

# Credit Card Fraud Detection

This project aims to detect fraudulent transactions in a credit card dataset using a machine learning model. The dataset used contains information about individual transactions, including various anonymized features, the transaction amount, and whether the transaction was fraudulent.

## Table of Contents

- [Problem Definition](#problem-definition)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building](#model-building)
- [Model Tuning](#model-tuning)
- [Documentation and Presentation](#documentation-and-presentation)

## Problem Definition

**Objective:** Build a model that predicts whether a credit card transaction is fraudulent or not.

**Questions to Answer:**
- What features of a transaction contribute to its likelihood of being fraudulent?
- How does the transaction amount differ between normal and fraudulent transactions?

## Data Collection

The dataset contains the following columns:
- Various anonymized features (V1, V2, ..., V28)
- `Amount`
- `Class` (target variable: 0 for normal, 1 for fraudulent)

```python
import pandas as pd

# Load the dataset
file_path = 'creditcard.csv'
file = pd.read_csv(file_path)

# Display the first few rows of the dataset
file.head(10)
```

## Data Preprocessing

- Handled missing values.
- Balanced the dataset by undersampling the majority class.
- Split the data into features and target variable.

```python
# Check for missing values
file.isnull().sum()

# Display value counts of the target variable
file['Class'].value_counts()

# Split the dataset into normal and fraudulent transactions
normal = file[file.Class == 0]
fraud = file[file.Class == 1]

print(normal.shape)
print(fraud.shape)

# Describe the 'Amount' feature for normal and fraudulent transactions
normal.Amount.describe()
fraud.Amount.describe()

# Sample the normal transactions to balance the dataset
normal_sample = normal.sample(n=492)
new_file = pd.concat([normal_sample, fraud], axis=0)

# Display the first few rows of the balanced dataset
new_file.head(10)

# Display value counts of the target variable in the balanced dataset
new_file['Class'].value_counts()

# Display the mean of each feature grouped by the target variable
new_file.groupby('Class').mean()

# Split the dataset into features and target variable
X = new_file.drop(columns='Class', axis=1)
Y = new_file['Class']
```

## Exploratory Data Analysis (EDA)

Visualized and analyzed the data to gain insights.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the distribution of the 'Amount' feature for normal and fraudulent transactions
plt.figure(figsize=(12, 6))
sns.histplot(normal['Amount'], bins=50, kde=True, color='blue', label='Normal')
sns.histplot(fraud['Amount'], bins=50, kde=True, color='red', label='Fraud')
plt.legend()
plt.title('Transaction Amount Distribution')
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(15, 10))
correlation_matrix = file.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Visualize the amount of transactions over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=file, x='Time', y='Amount', hue='Class', palette=['blue', 'red'])
plt.title('Transaction Amount Over Time')
plt.show()
```

## Model Building

Split the data into training and testing sets, then built and trained a model.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Predict and evaluate on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) * 100
print(f"Training Data Accuracy: {training_data_accuracy}%")

# Predict and evaluate on testing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) * 100
print(f"Test Data Accuracy: {test_data_accuracy}%")
```

## Model Tuning

Optimized the model by adjusting hyperparameters and using techniques like cross-validation.

```python
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, Y_train)

# Best parameters and model evaluation
best_model = grid_search.best_estimator_
Y_pred_best = best_model.predict(X_test)
best_test_data_accuracy = accuracy_score(Y_test, Y_pred_best) * 100
print(f"Best Test Data Accuracy: {best_test_data_accuracy}%")
print(classification_report(Y_test, Y_pred_best))
```

## Documentation and Presentation

- Documented the entire process, including problem definition, data collection, preprocessing, EDA, model building, and tuning.
- Created visualizations to support the findings and model evaluation.
- Presented the project and results effectively through clear and concise documentation.
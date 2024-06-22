# Iris Flower Classification

Author: Dhananjay Pant
Domain: Data Science
Batch: June 2025

This project aims to predict the species of Iris flowers based on their sepal and petal measurements using machine learning models. The Iris dataset contains measurements of 150 Iris flowers from three different species: setosa, versicolor, and virginica. The features in the dataset include sepal length, sepal width, petal length, and petal width.

## Table of Contents

- [Problem Definition](#problem-definition)
- [Data Collection](#data-collection)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Prediction](#prediction)

## Problem Definition

**Objective:** Build a model that can predict the species of Iris flowers based on their sepal and petal measurements.

**Questions to Answer:**
- How well can we predict the species of Iris flowers based on their measurements?
- Which features are most important for predicting the species?

## Data Collection

The Iris dataset is a popular dataset in machine learning and can be easily loaded using libraries like `scikit-learn`.

```python
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
```

## Data Exploration

```python
import pandas as pd

# Create a DataFrame from the Iris dataset
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Display the first few rows of the dataset
print(iris_df.head())

# Check the distribution of the target variable
print(iris_df['species'].value_counts())
```

## Data Preprocessing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Model Building

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Model Evaluation

The accuracy of the model can be evaluated using metrics like accuracy, precision, recall, or F1-score.

## Prediction

```python
# Predict the species of a new Iris flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Sepal length, Sepal width, Petal length, Petal width
new_flower_scaled = scaler.transform(new_flower)
prediction = knn.predict(new_flower_scaled)

# Map the prediction to the species name
species = iris.target_names[prediction[0]]
print("Predicted Species:", species)
```

---

Feel free to modify and expand upon this template to fit your project's specific requirements and details.
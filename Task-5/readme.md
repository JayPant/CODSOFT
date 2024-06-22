
# Movie Rating Prediction with Python

**Author:** Dhananjay Pant
**Domain:** Data Science  
**Date:** June 2024

## Overview

This project aims to predict the rating of a movie based on features like genre, director, and actors using a machine learning model. The dataset used contains information about individual movies, such as their name, year, duration, genre, rating, votes, director, and main actors.

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn

## Dataset

The dataset `creditcard.csv` contains the following columns:
- `Time`
- `Amount`
- `Class`

## Preprocessing

```python
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Preprocessing
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
X = data.drop(['Time', 'Class'], axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Handling Class Imbalance

```python
from imblearn.over_sampling import SMOTE

# Handling Class Imbalance
oversample = SMOTE()
X_train_balanced, y_train_balanced = oversample.fit_resample(X_train, y_train)
```

## Model Training

```python
from sklearn.ensemble import RandomForestClassifier

# Model Training
model = RandomForestClassifier()
model.fit(X_train_balanced, y_train_balanced)
```

## Model Evaluation

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Model Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## Results

The model achieved [insert your evaluation metrics here].

## Usage

To run the code, ensure you have the necessary libraries installed. You can run the provided code in a Jupyter notebook or any Python environment.


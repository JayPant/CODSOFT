# Titanic Survival Prediction
Author: Dhananjay Pant
Domain: Data Science
Batch: June2024

This project aims to predict whether a passenger on the Titanic survived or not using a machine learning model. The dataset used contains information about individual passengers, such as their age, gender, ticket class, fare, cabin, and whether or not they survived.

## Table of Contents

- [Problem Definition](#problem-definition)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building](#model-building)
- [Model Tuning](#model-tuning)
- [Deployment](#deployment)
- [Documentation and Presentation](#documentation-and-presentation)

## Problem Definition

**Objective:** Build a model that predicts whether a passenger on the Titanic survived or not.

**Questions to Answer:**
- What features of a passenger contribute to their survival?
- How does the survival rate vary across different passenger classes, genders and ages, etc.?

## Data Collection

The Titanic dataset was downloaded from Kaggle: [Titanic Dataset](https://www.kaggle.com/c/titanic/data).

```python
import pandas as pd

# Load the dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display the first few rows of the dataset
train_data.head()
```

## Data Preprocessing

- Handled missing values by filling with median/mean or dropping rows/columns.
- Encoded categorical variables such as gender and embarked.
- Normalized/scaled numerical features if necessary.

```python
# Check data for missing values
train_data.isnull().sum()

# Filling Missing Values of Age(177), Embarked(2) 
train_data['Age']= train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked']= train_data['Embarked'].fillna(train_data['Embarked'].mode())


# Encode categorical Variables- Sex and Embarked
train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
train_data['Embarked'] = train_data['Embarked'].map({'S':0, 'C':1, 'Q':2})

# Drop irrelevant features
train_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1, inplace=True)

```

## Exploratory Data Analysis (EDA)

Visualized and analyzed the data to gain insights.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Survival rate by gender
sns.barplot(x='Sex', y='Survived', data=train_data)
plt.title('Survival Rate by Gender')
plt.show()

# Survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Survival rate by age
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Age'][train_data['Survived'] == 1], bins=30, kde=False, label='Survived')
sns.histplot(train_data['Age'][train_data['Survived'] == 0], bins=30, kde=False, label='Not Survived')
plt.legend()
plt.title('Survival Rate by Age')
plt.show()
```

## Model Building

Split the data into training and testing sets, then built and trained a model.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split the data
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Model Tuning

Optimized the model by adjusting hyperparameters and using techniques like cross-validation.

```python
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and model evaluation
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print('Best Accuracy:', accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))
```



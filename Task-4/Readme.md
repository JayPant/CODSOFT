# Sales Prediction Using Python

**Author:** Dhananjay Pant  
**Domain:** Data Science  
**Date:** JUNE BATCH A57 

Sales prediction involves forecasting the amount of a product that customers will purchase, taking into account various factors such as advertising expenditure, target audience segmentation, and advertising platform selection. In businesses that offer products or services, the role of a Data Scientist is crucial for predicting future sales. They utilize machine learning techniques in Python to analyze and interpret data, allowing them to make informed decisions regarding advertising costs. By leveraging these predictions, businesses can optimize their advertising strategies and maximize sales potential. Let's embark on the journey of sales prediction using machine learning in Python.

## Problem Definition

**Objective:** Build a machine learning model to predict sales based on advertising expenditure on TV, Radio, and Newspaper.

## Data Collection

The dataset contains the following columns:
- TV: Advertising expenditure on TV
- Radio: Advertising expenditure on Radio
- Newspaper: Advertising expenditure on Newspaper
- Sales: Sales generated

```python
# Load the dataset
import pandas as pd

file_path = 'sales_data.csv'
sales_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(sales_data.head())
```

## Exploratory Data Analysis (EDA)

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot to visualize the relationship between features and target
sns.pairplot(sales_data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.7)
plt.show()
```

## Data Preprocessing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data into features (X) and target (y)
X = sales_data[['TV', 'Radio', 'Newspaper']]
y = sales_data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Model Building

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
```

## Prediction

```python
# Predict sales for a new set of advertising expenditures
new_data = [[100, 50, 25]]  # TV, Radio, Newspaper
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

print("Predicted Sales:", prediction[0])
```

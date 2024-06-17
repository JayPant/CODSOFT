# Movie Rating Prediction with Python

This project aims to predict the rating of a movie based on features like genre, director, and actors using a machine learning model. The dataset used contains information about individual movies, such as their name, year, duration, genre, rating, votes, director, and main actors.

## Table of Contents

- [Problem Definition](#problem-definition)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building](#model-building)
- [Model Tuning](#model-tuning)
- [Prediction with New Data](#prediction-with-new-data)


## Problem Definition

**Objective:** Build a model that predicts the rating of a movie based on features like genre, director, and actors.

**Questions to Answer:**
- What features of a movie contribute to its rating?
- How do the ratings vary across different genres, directors, and actors?

## Data Collection

The dataset contains the following columns:
- `Name`
- `Year`
- `Duration`
- `Genre`
- `Rating`
- `Votes`
- `Director`
- `Actor 1`
- `Actor 2`
- `Actor 3`

```python
import pandas as pd

# Load the dataset
file_path = 'path/to/your/IMDb_Movies_India.csv'
movie_data = pd.read_csv(file_path, encoding="latin1")

# Display the first few rows of the dataset
movie_data.head()
```

## Data Preprocessing

- Handled missing values by filling with median/mean or dropping rows/columns.
- Converted categorical variables such as genre, director, and actors into numerical format.
- Normalized/scaled numerical features if necessary.

```python
# Check for missing values
movie_data.isnull().sum()

# Drop rows with missing 'Rating'
movie_data.dropna(subset=["Rating"], inplace=True)

# Drop rows with missing values in critical columns
movie_data.dropna(subset=['Actor 1', 'Actor 2', 'Actor 3', 'Director', 'Genre'], inplace=True)


from sklearn.preprocessing import StandardScaler


# Clean 'Votes' column
movie_data['Votes'] = movie_data['Votes'].astype(str).str.replace(',', '', regex=False)
movie_data['Votes'] = pd.to_numeric(movie_data['Votes'], errors='coerce')

# Clean 'Duration' column
movie_data['Duration'] = movie_data['Duration'].astype(str).str.replace('min', '', regex=False)
movie_data['Duration'] = pd.to_numeric(movie_data['Duration'], errors='coerce')
movie_data['Duration'].fillna(movie_data['Duration'].median(), inplace=True)

# One-Hot Encoding for categorical variables
movie_data = pd.get_dummies(movie_data, columns=['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
movie_data[['Duration', 'Votes']] = scaler.fit_transform(movie_data[['Duration', 'Votes']])

# Check for missing values again
print("\nMissing values after preprocessing:")
print(movie_data.isnull().sum())

```

## Exploratory Data Analysis (EDA)

Visualized and analyzed the data to gain insights.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Rating distribution
sns.histplot(movie_data['Rating'], bins=30, kde=True)
plt.title('Rating Distribution')
plt.show()

# Rating by genre
plt.figure(figsize=(14, 8))
sns.boxplot(x='Genre', y='Rating', data=movie_data)
plt.title('Rating by Genre')
plt.xticks(rotation=90)
plt.show()

# Rating by director (show top 10 directors by movie count)
top_directors = movie_data['Director'].value_counts().head(10).index
top_directors_data = movie_data[movie_data['Director'].isin(top_directors)]
plt.figure(figsize=(14, 8))
sns.boxplot(x='Director', y='Rating', data=top_directors_data)
plt.title('Rating by Director')
plt.xticks(rotation=90)
plt.show()
```

## Model Building

Split the data into training and testing sets, then built and trained a model.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define features and target
X = movie_data.drop(['Name', 'Year', 'Rating'], axis=1)
y = movie_data['Rating']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
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
print('Best Mean Squared Error:', mean_squared_error(y_test, y_pred_best))
```

## Prediction with New Data

```python
# Example new movie data (make sure to include all necessary columns with 0s where appropriate)
new_movie = {
    'Duration': [120],  # example duration
    'Votes': [1000],    # example number of votes
    # One-hot encoded genre (adjust column names as per your dataset)
    'Genre_Action': [1],  
    'Genre_Comedy': [0],
    'Genre_Drama': [0],
    # One-hot encoded director (adjust column names as per your dataset)
    'Director_Some Director': [1],  
    # One-hot encoded actors (adjust column names as per your dataset)
    'Actor 1_Some Actor': [1],      
    'Actor 2_Some Actor': [0],      
    'Actor 3_Some Actor': [0]       
}

# Convert new movie data to DataFrame
new_movie_df = pd.DataFrame(new_movie)

# Ensure the new data has the same columns as the training data
new_movie_df = new_movie_df.reindex(columns=X.columns, fill_value=0)

# Normalize numerical features
new_movie_df[['Duration', 'Votes']] = scaler.transform(new_movie_df[['Duration', 'Votes']])

# Predict the rating
predicted_rating = model.predict(new_movie_df)
print('Predicted Rating:', predicted_rating)
```


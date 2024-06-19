# Movie Rating Prediction with Python

This project aims to predict the rating of a movie based on features like genre, director, and actors using a machine learning model. The dataset used contains information about individual movies, such as their name, year, duration, genre, rating, votes, director, and main actors.

## Table of Contents

- [Problem Definition](#problem-definition)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
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
# Load the dataset
file_path = 'moviesdata.csv'
imdb_df = pd.read_csv(file_path, encoding="latin1")
# Display the first few rows of the dataset

```

## Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# Drop unnecessary columns
imdb_df.drop(['Name', 'Unnamed: 0'], axis=1, inplace=True)

# Handle missing values
imdb_df.dropna(inplace=True)

# Convert categorical variables to numerical
imdb_df = pd.get_dummies(imdb_df, columns=['Genre'], drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
imdb_df[['Year', 'Votes', 'Duration']] = scaler.fit_transform(imdb_df[['Year', 'Votes', 'Duration']])
```

## Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Rating distribution
sns.histplot(imdb_df['Rating'], bins=30, kde=True)
plt.title('Rating Distribution')
plt.show()

# Genre vs. Rating
plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Rating', data=imdb_df)
plt.title('Genre vs. Rating')
plt.xticks(rotation=45)
plt.show()
```

## Model Building

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define features and target variable
X = imdb_df.drop('Rating', axis=1)
y = imdb_df['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict ratings
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
```

## Model Evaluation

```python
print('Mean Squared Error:', mse)
```

## Prediction with New Data

```python
# Example new movie data
new_movie = {
    'Year': [2023],
    'Votes': [5000],
    'Duration': [120],
    'Genre_Action': [1],
    'Genre_Comedy': [0],
    'Genre_Drama': [0]
}

# Convert to DataFrame and normalize
new_movie_df = pd.DataFrame(new_movie)
new_movie_df[['Year', 'Votes', 'Duration']] = scaler.transform(new_movie_df[['Year', 'Votes', 'Duration']])

# Predict rating for the new movie
predicted_rating = model.predict(new_movie_df)
print('Predicted Rating:', predicted_rating)
```

```

markdown
# Movie Rating Prediction with Python

This project aims to predict the rating of a movie based on features like genre, director, and actors using a machine learning model. The dataset used contains information about individual movies, such as their name, year, duration, genre, rating, votes, director, and main actors.

## Data Preprocessing

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = 'moviesdata.csv'
imdb_df = pd.read_csv(file_path, encoding="latin1")

# Drop unnecessary columns
imdb_df.drop(['Name', 'Unnamed: 0'], axis=1, inplace=True)

# Handle missing values
imdb_df.dropna(inplace=True)

# Convert categorical variables to numerical
imdb_df = pd.get_dummies(imdb_df, columns=['Genre'], drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
imdb_df[['Year', 'Votes', 'Duration']] = scaler.fit_transform(imdb_df[['Year', 'Votes', 'Duration']])
```

## Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Rating distribution
sns.histplot(imdb_df['Rating'], bins=30, kde=True)
plt.title('Rating Distribution')
plt.show()

# Genre vs. Rating
plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Rating', data=imdb_df)
plt.title('Genre vs. Rating')
plt.xticks(rotation=45)
plt.show()
```

## Model Building

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define features and target variable
X = imdb_df.drop('Rating', axis=1)
y = imdb_df['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict ratings
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
```

## Model Evaluation

```python
print('Mean Squared Error:', mse)
```

## Prediction with New Data

```python
# Example new movie data
new_movie = {
    'Year': [2023],
    'Votes': [5000],
    'Duration': [120],
    'Genre_Action': [1],
    'Genre_Comedy': [0],
    'Genre_Drama': [0]
}

# Convert to DataFrame and normalize
new_movie_df = pd.DataFrame(new_movie)
new_movie_df[['Year', 'Votes', 'Duration']] = scaler.transform(new_movie_df[['Year', 'Votes', 'Duration']])

# Predict rating for the new movie
predicted_rating = model.predict(new_movie_df)
print('Predicted Rating:', predicted_rating)
```


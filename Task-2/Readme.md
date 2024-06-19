# Movie Rating Prediction with Python

This project aims to predict the rating of a movie based on features like genre, director, and actors using a machine learning model. The dataset used contains information about individual movies, such as their name, year, duration, genre, rating, votes, director, and main actors.

## Table of Contents

- [Problem Definition](#problem-definition)
- [Data Collection](#data-collection)
- [Insights of data](#insights-of-data)
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
# Import Libraries for data processing and modeling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
file_path = 'moviesdata.csv'
movies = pd.read_csv(file_path, encoding="latin1")
# Display the first few rows of the dataset

print(movies.head())
movies.shape
movies.info()
```


## Insights of Data

```python

print(movies.duplicated().sum())
print()
movies.dropna(inplace=True)
print(movies.shape)

movies.isnull().sum()
movies.drop_duplicates(inplace=True)
movies.columns
```

## Data Preprocessing

```python
# Replacing the brackets from year column

import re

# Function to extract numeric part from a string
def extract_numeric(s):
    match = re.search(r'\d+', str(s))
    if match:
        return int(match.group())
    return None

# Apply the function to extract numeric part from 'Year' column
movies['Year'] = movies['Year'].apply(extract_numeric)

# Remove the min word from 'Duration' column and convert all values to numeric
movies['Duration'] = pd.to_numeric(movies['Duration'].astype(str).str.replace(' min', ''), errors='coerce')

 

# Splitting the genre by, to keep only unique genres and replacing the null values with mode
movies['Genre'] = movies['Genre'].str.split(', ')
movies = movies.explode('Genre')
movies['Genre'].fillna(movies['Genre'].mode()[0], inplace=True)    


# Convert 'Votes' to numeric and replace the , to keep only numerical part
movies['Votes'] = pd.to_numeric(movies['Votes'].astype(str).str.replace(',', ''), errors='coerce')


# Checking the dataset is there any null values present and data types of the features present
movies.info()
```

## Exploratory Data Analysis (EDA)

```python
# Here we have created a histogram over the years in the data
year = px.histogram(movies, x='Year', histnorm='probability density', nbins=30)
year.show()

# Group data by Year and calculate the average rating
avg_rating_by_year = movies.groupby(['Year', 'Genre'])['Rating'].mean().reset_index()

# Top 10 genres
top_genres = movies['Genre'].value_counts().head(10).index

# Top 3 genres
average_rating_by_year = avg_rating_by_year[avg_rating_by_year['Genre'].isin(top_genres)]

# Line plot with Plotly Express
fig = px.line(average_rating_by_year, x='Year', y='Rating', color="Genre")
fig.update_layout(title='Average Rating by Year for Top Genres', xaxis_title='Year', yaxis_title='Average Rating')

# Show the plot
fig.show()


# This histogram shows the distribution of ratings and its probable density
rating_fig = px.histogram(movies, x='Rating', histnorm='probability density', nbins=40)
rating_fig.update_layout(title='Distribution of Rating', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Rating', yaxis_title='Probability Density', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), bargap=0.02, plot_bgcolor='white')
rating_fig.show()
```

## Model Building

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

# Dropping Name 
movies.drop('Name', axis=1, inplace=True)

# Grouping the columns with their average rating and then creating a new feature
genre_mean_rating = movies.groupby('Genre')['Rating'].transform('mean')
movies['Genre_mean_rating'] = genre_mean_rating

director_mean_rating = movies.groupby('Director')['Rating'].transform('mean')
movies['Director_encoded'] = director_mean_rating

actor1_mean_rating = movies.groupby('Actor 1')['Rating'].transform('mean')
movies['Actor1_encoded'] = actor1_mean_rating

actor2_mean_rating = movies.groupby('Actor 2')['Rating'].transform('mean')
movies['Actor2_encoded'] = actor2_mean_rating

actor3_mean_rating = movies.groupby('Actor 3')['Rating'].transform('mean')
movies['Actor3_encoded'] = actor3_mean_rating


# SPlit train and test data
X = movies[['Year', 'Votes', 'Duration', 'Genre_mean_rating', 'Director_encoded', 'Actor1_encoded', 'Actor2_encoded', 'Actor3_encoded']]
y = movies['Rating']

# Splitting the dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building machine learning model and training them
Model = LinearRegression()
Model.fit(X_train, y_train)
Model_pred = Model.predict(X_test)

# Evaluating the performance of the model with evaluation metrics
print('The performance evaluation of Logistic Regression is below: ', '\n')
print('Mean squared error: ', mean_squared_error(y_test, Model_pred))
print('Mean absolute error: ', mean_absolute_error(y_test, Model_pred))
print('R2 score: ', r2_score(y_test, Model_pred))
```

## Model Evaluation 

```python

# Evaluating the performance of the model with evaluation metrics
print('The performance evaluation of Logistic Regression is below: ', '\n')
print('Mean squared error: ', mean_squared_error(y_test, Model_pred))
print('Mean absolute error: ', mean_absolute_error(y_test, Model_pred))
print('R2 score: ', r2_score(y_test, Model_pred))
```

## Prediction with New Data

```python
# For testing, We create a new dataframe with values close to the any of our existing data to evaluate
data = {'Year': [2015], 'Votes': [58], 'Duration': [120], 'Genre_mean_rating': [7.8], 'Director_encoded': [3.5], 'Actor1_encoded': [5.3], 'Actor2_encoded': [4.5], 'Actor3_encoded': [4.5]}
trail = pd.DataFrame(data)

# Predict the movie rating by entered data
rating_predicted = Model.predict(trail)

# Display the predicted result from the Model
print("Predicted Rating:", rating_predicted[0])
```




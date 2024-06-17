Certainly! Here is the modified code for the `Movie Rating Prediction with Python` project, incorporating your provided reference for handling the dataset:

```python
# Movie Rating Prediction with Python

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

## Data Collection

# Load the dataset
file_path = 'path/to/your/dataset.csv'
movies = pd.read_csv(file_path, encoding="latin1")

# Display the first few rows of the dataset
print(movies.head())

# Display dataset information
print(movies.info())

# Display dataset statistics
print(movies.describe())

## Data Preprocessing

# Check for missing values
print(movies.isnull().sum())

# Drop rows where 'Rating' is missing
movies.dropna(subset=["Rating"], inplace=True)

# Drop rows where 'Actor 1', 'Actor 2', 'Actor 3', 'Director', or 'Genre' are missing
movies.dropna(subset=['Actor 1','Actor 2','Actor 3','Director','Genre'], inplace=True)

# Check for missing values again
print(movies.isnull().sum())

# Convert 'Votes' column to integer
movies['Votes'] = movies['Votes'].str.replace(',', '').astype(int)

# Convert 'Year' column to integer
movies['Year'] = movies['Year'].str.strip('()').astype(int)

# Convert 'Duration' column to numeric (extract numeric part and convert to float)
movies['Duration'] = movies['Duration'].str.extract('(\d+)').astype(float)

# Fill missing values in 'Duration' with the median value
movies['Duration'].fillna(movies['Duration'].median(), inplace=True)

# Check for missing values again
print(movies.isnull().sum())

# One-Hot Encoding for categorical variables
movies = pd.get_dummies(movies, columns=['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
movies[['Duration', 'Votes']] = scaler.fit_transform(movies[['Duration', 'Votes']])

## Exploratory Data Analysis (EDA)

# Rating distribution
sns.histplot(movies['Rating'], bins=30, kde=True)
plt.title('Rating Distribution')
plt.show()

# Rating by genre
plt.figure(figsize=(14, 8))
sns.boxplot(x='Genre', y='Rating', data=movies)
plt.title('Rating by Genre')
plt.xticks(rotation=90)
plt.show()

# Rating by director (show top 10 directors by movie count)
top_directors = movies['Director'].value_counts().head(10).index
top_directors_data = movies[movies['Director'].isin(top_directors)]
plt.figure(figsize=(14, 8))
sns.boxplot(x='Director', y='Rating', data=top_directors_data)
plt.title('Rating by Director')
plt.xticks(rotation=90)
plt.show()

## Model Building

# Define features and target
X = movies.drop(['Name', 'Year', 'Rating'], axis=1)
y = movies['Rating']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

## Prediction with New Data

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


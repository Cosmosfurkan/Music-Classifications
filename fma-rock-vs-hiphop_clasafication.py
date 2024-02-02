# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sbs
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Reading the dataset
tracks = pd.read_csv("Veri setleri/archive/rock-vs-hiphop.csv")

# Reading echonest metrics and merging with the main dataset
echonest_metrics = pd.read_json("Veri setleri/archive/echonest-metric.json", precise_float=True)
echo_tracks = echonest_metrics.merge(tracks[['track_id', 'genre_top']], on='track_id')
print(echo_tracks.head())

# Displaying missing values in the dataset
print(tracks.isnull().sum())

# Handling missing values using SimpleImputer
imputer = SimpleImputer()
imputed = imputer.fit_transform(echo_tracks.drop('genre_top', axis=1))  # NaN values are replaced with mean values

# Label Encoding for the target variable 'genre_top'
le = LabelEncoder()
echo_tracks['genre_top'] = le.fit_transform(echo_tracks['genre_top'])

# Renaming the column 'genre_top' to 'label'
echo_tracks.rename(columns={'genre_top': 'label'}, inplace=True)

# Splitting the dataset into train and test sets
X = echo_tracks.drop(["label", "track_id"], axis=1)
y = echo_tracks["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Logistic Regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Evaluating Logistic Regression model accuracy
lr_accuracy = logreg.score(X_test, y_test)
print(f"Logistic Regression Accuracy: {lr_accuracy}")

# Training Decision Tree model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Evaluating Decision Tree model accuracy
dt_accuracy = dt.score(X_test, y_test)
print(f"Decision Tree Accuracy: {dt_accuracy}")

# Handling class imbalances by resampling
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score

data_balanced = pd.concat([resample(echo_tracks[echo_tracks['label'] == 0], n_samples=1000),
                           resample(echo_tracks[echo_tracks['label'] == 1], n_samples=1000)])

# Splitting the resampled data into training and validation sets
X_balanced = data_balanced.drop(["label", "track_id"], axis=1)
y_balanced = data_balanced['label']
X_train_balanced, X_valid_balanced, y_train_balanced, y_valid_balanced = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Evaluating model performance using cross-validation
cv_scores_lr = cross_val_score(logreg, X_balanced, y_balanced, cv=5)
cv_scores_dt = cross_val_score(dt, X_balanced, y_balanced, cv=5)

print(f"Logistic Regression Cross-Validation Scores: {cv_scores_lr}")
print(f"Decision Tree Cross-Validation Scores: {cv_scores_dt}")

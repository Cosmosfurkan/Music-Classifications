# Music Genre Classification

This repository aims to build a music genre classification model using machine learning techniques. The dataset includes audio track features from the Echo Nest and genre labels indicating whether the track belongs to the 'Rock' or 'Hip-Hop' genre.

## Dataset

The primary dataset is loaded from the file "rock-vs-hiphop.csv" located in the "Veri setleri/archive" directory. Echonest metrics are extracted from the file "echonest-metric.json" and merged with the main dataset based on the 'track_id' column.

## Data Preprocessing

### Handling Missing Values

Check and handle missing values in the dataset using the `SimpleImputer` class.

### Label Encoding

Encode the target variable 'genre_top' using Label Encoding.

### Train-Test Split

Split the dataset into training and testing sets.

## Model Training and Evaluation

### Logistic Regression Model

Train a Logistic Regression model and evaluate its accuracy.

### Decision Tree Model

Train a Decision Tree model and evaluate its accuracy.

## Handling Class Imbalance

Resample the dataset to address class imbalances and perform cross-validation.

## Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/music-genre-classification.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:

   ```bash
   python main.py
   ```

Feel free to explore and enhance the code for further improvements in model performance.

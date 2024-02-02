import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sbs
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# veri setini okuyoruz

tracks = pd.read_csv("Veri setleri/archive/rock-vs-hiphop.csv")

echonest_metrics = pd.read_json("Veri setleri/archive/echonest-metric.json" , precise_float=True) # echonest metriclerini okuruz
echo_tracks = echonest_metrics.merge(tracks[['track_id','genre_top']],on='track_id')
print(echo_tracks.head())
#print(tracks["tags"].head())
# Veri setindeki eksik verileri dolduruyoruz.
print(tracks.isnull().sum()) # Eksik değer sayılarını gösterir.

# Eksik verileri doldurmak için Imputer sınıfını kullanıyoruz.
imputer = SimpleImputer()
imputed = imputer.fit_transform(echo_tracks.drop('genre_top', axis=1))  # NaN values are replaced with mean values

#Label Encoding yaptırıyoruz.
le = LabelEncoder()
echo_tracks['genre_top'] = le.fit_transform(echo_tracks['genre_top'])

# Sütun ismini degistiriyoruz.
echo_tracks.rename(columns={'genre_top':'label'},inplace=True)

# Veri setini train ve test olarak ayırıyoruz.
X = echo_tracks.drop(["label","track_id"],axis= 1)
y = echo_tracks["label"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

lr_accuracy = logreg.score(X_test, y_test)
print(f"Logistic Regression Accuracy: {lr_accuracy}")

# model DesecionTree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

dt_accuracy = dt.score(X_test, y_test)
print(f"DesecionTree Accuracy: {dt_accuracy}")

# veri setindeki sınıflar arası  dengesizlikler

from sklearn.utils import resample
from sklearn.model_selection import cross_val_score

data_balanced = pd.concat([resample(echo_tracks[echo_tracks['label'] == 0], n_samples=1000),
                           resample(echo_tracks[echo_tracks['label'] == 1], n_samples=1000)])

# Yeniden bölme sonrası eğitim ve doğrulama setlerini oluştur
X_balanced = data_balanced.drop(["label","track_id"], axis=1)
y_balanced = data_balanced['label']
X_train_balanced, X_valid_balanced, y_train_balanced, y_valid_balanced = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Cross-validation kullanarak model performansını değerlendir
cv_scores_lr = cross_val_score(logreg, X_balanced, y_balanced, cv=5)
cv_scores_dt = cross_val_score(dt, X_balanced, y_balanced, cv=5)

print(f"Logistic Regression Cross-Validation Scores: {cv_scores_lr}")
print(f"Decision Tree Cross-Validation Scores: {cv_scores_dt}")
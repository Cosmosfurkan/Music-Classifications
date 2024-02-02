import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sbs
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
#import keras as keras
import matplotlib.pyplot as plt
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout
#from keras.optimizers import SGD
#from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split


data = pd.read_csv("C:/Users/furkan/Desktop/Yapay zeka/Veri setleri/ParisHousing.csv")
#print(data.describe())

# 1.1) Data Preprocessing
data = data.drop_duplicates() # tekrarlanan satırları silmek için kullanılır
data = data.reset_index(drop=True)
data = data.dropna(axis=0) # eksik verileri silmek için kullanılır
data = data.reset_index(drop=True)
data = data.drop(data[data.price < 1000].index)
data = data.reset_index(drop=True)
data = data.drop(data[data.price > 1000000].index)
data = data.reset_index(drop=True)
data = data.drop(data[data.numberOfRooms > 10].index)
data = data.reset_index(drop=True)
data = data.drop(data[data.numberOfRooms < 1].index)
data = data.reset_index(drop=True)
data = data.drop(data[data.floors > 35].index)
data = data.reset_index(drop=True)
data = data.drop(data[data.floors < 0].index)
data = data.reset_index(drop=True)

data_square = data["squareMeters"]
data_price = data["price"]
data_rooms = data["numberOfRooms"]
data_floors = data["floors"]
data_attic = data["attic"]
data_basement = data["basement"]
data_citycode = data["cityCode"]
data_garage = data["garage"]
data_lift = data["hasYard"]
"""
print(sns.boxplot(x = data_price))
plt.show()
print(sns.boxplot(x = data_square))
plt.show()
print(sns.boxplot(x = data_rooms))
plt.show()
print(sns.boxplot(x = data_floors))
plt.show()
"""
# 1.2) Data Visualization
#print(sns.pairplot(data["price"]))
"""
data = pd.DataFrame({
    "price": data_price,
    "square": data_square,
    "rooms": data_rooms,
    "floors": data_floors
})
"""
#print(sns.pairplot(data))
#plt.show()

# 1.3) Data Normalization
data = (data - data.min()) / (data.max() - data.min())
print(data.head())

# 2) Model Training and Prediction

# 2.1) Data Splitting
train_data = data.sample(frac=0.8, random_state=200)
test_data = data.drop(train_data.index)
# 2.2) Feature Selection
train_data_features = train_data.copy()
test_data_features = test_data.copy()
train_data_features.pop("price")
test_data_features.pop("price")
# 2.3) Label Selection
train_data_labels = train_data["price"]
test_data_labels = test_data["price"]
# 2.4) Model Selection
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=16))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='softmax'))
model.add(Dense(1, kernel_initializer="glorot_uniform", activation='softmax'))
# Compile the model with Mean Square Error as loss function and Adam as optimizer
model.add(Dense(1, activation='sigmoid'))
# 2.5) Model Compilation
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 2.6) Model Training
model.fit(train_data_features, train_data_labels, epochs=20, batch_size=32)
# 2.7) Model Prediction
predictions = model.predict(test_data_features)
# 2.8) Model Evaluation
score = model.evaluate(test_data_features, test_data_labels, batch_size=128)
print(score)


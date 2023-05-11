import sklearn 
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()
#grab entire buying column and transform into the integer values
buying = le.fit_transform(list(data["buying"]))
main = le.fit_transform(list(data["main"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
saftey = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))


predict = "class"

X = list(zip(buying,main,door,persons,lug_boot,saftey))
y = list(cls)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

print(X_train, y_test)
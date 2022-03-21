import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import pickle
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
x = mnist['data']
y = mnist['target']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
x_train=x_train/255
x_test=x_test/255
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train,y_train)
from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print("Accuracy of the model is {0}%".format(round(accuracy_score(y_test,y_pred)*100,2)))
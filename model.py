import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import joblib
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
x = mnist['data']
y = mnist['target']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
x_train=x_train/255
x_test=x_test/255
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
svm = SVC(kernel = 'poly',degree = 3)
svm.fit(x_train,y_train)
svm.score(x_test,y_test)
# model = RandomForestClassifier()
# model.fit(x_train,y_train)
joblib.dump(svm,open('model_SVM.pkl','wb'), compress=9)
from sklearn.metrics import accuracy_score
# y_pred = model.predict(x_test)
# print("Accuracy of the model is {0}%".format(round(accuracy_score(y_test,y_pred)*100,2)))
import pandas as pd  
import numpy as np
import sklearn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
# Loading Dataset to Panda
features = pd.read_csv('Datasetnotes.csv')
features = features.sample(frac = 1).reset_index(drop=True)
# Getting all column except 'Class'
X = features.drop('Class', axis=1)
# Getting column named 'Class'
y = features['Class']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

print("XTrain")
print(X_train.shape)
print("XTest")
print(X_test.shape)
#Trainer
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
# Serializing model to file
pickle.dump(svclassifier,open("model.pkl","wb"))
y_pred = svclassifier.predict(X_test)

#print(svclassifier.score(X_test, y_test))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


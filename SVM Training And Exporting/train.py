import pandas as pd  
import numpy as np
import sklearn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
# Loading Dataset to Panda
directory = '../Training Dataset/'
features = pd.read_csv(directory + 'Datasetnotes.csv')
features = features.sample(frac = 1).reset_index(drop=True)
# Getting all column except 'Class'
X = features.drop('Class', axis=1)
# Getting column named 'Class'
y = features['Class']

svclassifier = SVC(kernel='linear')
svclassifier.fit(X,y)
# Serializing model to file
pickle.dump(svclassifier,open("model.pkl","wb"))


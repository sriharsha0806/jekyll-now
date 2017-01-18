# Decision Trees
Example using Decision Tree
>>> from sklearn import tree
>>> X = [[0,0],[1,1]]
>>> Y = [0,1]
>>> clf = tree.DecisionTreeClassifier()
>>> clf = clf.fit(X,Y)
>>> clf.predict([[2., 2.]])

# Decision Tree
import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData
from matplotlib import pyplot as plt
import pylab as pl
from sklearn import tree
def classify(features_train, labels_train):
  clf = tree.DecisionTreeClassifier()
  clf = clf.fit(features_train,labels_train)
  return clf

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = classify(features_train, labels_train)
preetyPicture(clf, features_test, labels_test)
output_image("test.png", "png" ,open("test.png","rb").read())

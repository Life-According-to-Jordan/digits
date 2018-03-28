from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
import numpy as np

#import dataset in to program
digits=datasets.load_digits()

#we have 1797 records

#each entry is 8*8 matrix, we shall reshape this matrix to 1*64 so as to make our computation more human readable

digits.images=digits.images.reshape(digits.images.shape[0],digits.images.shape[1]*digits.images.shape[2])

#now lets print new matrix dimensions
print(digits.images.shape)

#lets try to print features and labels we have in the dataset
print(digits.images)
print(digits.target)

#lets also see the dimension of data we have
print(digits.images.shape)
print(digits.target.shape)

#now we have data, we need to split this whole chunk of data in to training and testing data set
#the split is 75% training data and 25% testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(digits.images,digits.target,test_size=0.25)

#adding classifiers 
clf_KNN=KNeighborsClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_tree = tree.DecisionTreeClassifier()

#train classifiers
clf_tree.fit(x_train, y_train)
clf_svm.fit(x_train, y_train)
clf_perceptron.fit(x_train, y_train)
clf_KNN.fit(x_train, y_train)

# Testing using the same data
pred_tree = clf_tree.predict(x_test)
acc_tree = accuracy_score(y_test, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(x_test)
acc_svm = accuracy_score(y_test, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

pred_per = clf_perceptron.predict(x_test)
acc_per = accuracy_score(y_test, pred_per) * 100
print('Accuracy for perceptron: {}'.format(acc_per))

pred_KNN = clf_KNN.predict(x_test)
acc_KNN = accuracy_score(y_test, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

# The best classifier from svm, per, KNN
index = np.argmax([acc_tree, acc_svm, acc_per, acc_KNN])
classifiers = {0: 'Tree', 1: 'SVM', 2: 'Perceptron', 3: 'KNN'}
print('Best species classifier is {}'.format(classifiers[index]))
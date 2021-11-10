import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

dataMat = np.load('TestMatrices/dataMat.npy')
labelMat = np.load('TestMatrices/labelMat.npy')
#dataMat = np.load('testPDNF/dataMat.npy')
#labelMat = np.load('testPDNF/labelMat.npy')
print(dataMat.shape)
#vMat = np.load('testFD/vMat.npy')
#nvMat = np.load('testFD/nvMat.npy')
x_train,x_test,y_train,y_test=train_test_split(dataMat,labelMat,train_size=0.8,)
#x_train = np.load('testVDD/x_train.npy')
#x_test = np.load('testVDD/x_test.npy')
#y_train = np.load('testVDD/y_train.npy')
#y_test = np.load('testVDD/y_test.npy')

#print(y_train.shape)
#print(y_test.shape)
sx1 = len(x_train)
sx2= len(x_test)

x1=np.reshape(x_train,(sx1,2600))
x2= np.reshape(x_test,(sx2,2600))
print(x_train.shape)
print(x_test.shape)

#print(labelMat)
#clf = svm.SVC()
#clf = RandomForestClassifier()
clf = AdaBoostClassifier()
clf.fit(x1, y_train)
b=clf.score(x1,y_train)
c = clf.score(x2,y_test)
#print(a)
print(b)
print(c)

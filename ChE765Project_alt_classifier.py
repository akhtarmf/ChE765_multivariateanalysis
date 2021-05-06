import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_excel('tsne_output.xls')

X = data.drop('Label', axis=1)
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# SVM polynomial kernel

svc = SVC(kernel='poly', degree=10)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print('\nPolynomial Kernel')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SVM sigmoid kernel

svc = SVC(kernel='sigmoid')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print('\nSigmoid Kernel')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SVM Guassian kernel

svc = SVC(kernel='rbf', gamma=10)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print('\nRBF Kernel')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.show()
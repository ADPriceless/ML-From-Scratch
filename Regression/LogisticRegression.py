'''A program to practice logistic regression. The algorithm's
style is inspired by the SKLearn module'''

import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.datasets import make_classification
import numpy as np
import math

# Functions & class(es)

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def cross_entropy(prediction, label):
    return -(label*np.log10(prediction) + \
        (1-label)*np.log10(1-prediction))

class LogisticRegression():
    def __init__(self, alpha=0.01):
        self.ALPHA = alpha
        self.errors = []

    def train(self, train_data, train_labels):
        bias = -1*np.ones([train_data[:,0].size, 1])
        inputs = np.concatenate((train_data, bias), axis=1)
        self.w = 2*np.random.random(inputs[0].shape)-1
        
        for x, y in zip(inputs, train_labels):
            # Update weights
            output = sigmoid(np.dot(self.w, x))
            # error = cross_entropy(output, y)
            error = output-y
            delta_w = self.ALPHA*error*x
            self.w -= delta_w

            # Record error
            self.errors.append(error)

    def predict(self, test_data):
        predictions = []
        bias = -1*np.ones([test_data[:,0].size, 1])
        inputs = np.concatenate((test_data, bias), axis=1)
        for x in inputs:
            output = sigmoid(np.dot(self.w, x))
            p = 1 if output > 0.5 else 0
            predictions = np.append(predictions, p)
        return predictions

    def score(self, predictions, true_labels):
        count = 0
        for p, y in zip(predictions, true_labels):
            if p == y:
                count += 1
        return count / len(predictions)

    def plot_decision_boundary(self, test_data):
        '''x0 = -(w0/w1).x1+t/w1'''
        min_x0 = np.min(test_data[:,0])
        max_x0 = np.max(test_data[:,0])
        x0 = np.arange(min_x0, max_x0, 0.1*(max_x0-min_x0)) # 10 steps in x-axis
        x1 = -(self.w[0]*x0 + self.w[2]*(-1))/self.w[1]
        return x0, x1

def split_data(data, labels, test_ratio=0.2):
    test_size = int(np.floor(data[:,0].size*test_ratio))
    train_data = data[:-test_size,:]
    test_data = data[-test_size:,:]
    train_labels = labels[:-test_size]
    test_labels = labels[-test_size:]
    return train_data, test_data, train_labels, test_labels

def plot_test_data(test_data, test_labels):
    blue_class = []
    red_class = []
    for x, y in zip(test_data, test_labels):
        if y == 0:
            blue_class.append(x)
        else:
            red_class.append(x)
    blue_class = np.array(blue_class)
    red_class = np.array(red_class)
    return blue_class, red_class

# Algorithm

X, y = make_classification(
    n_samples=10000, 
    n_features=2, 
    n_classes=2, 
    n_clusters_per_class=1, 
    n_informative=2, 
    n_redundant=0,
    class_sep=1.5)
X_train, X_test, y_train, y_test = split_data(X, y)

clf = LogisticRegression(alpha=0.001)
clf.train(X_train, y_train)
y_pred = clf.predict(X_test)
print('Score:', clf.score(y_pred, y_test))

fig_clf = plt.figure(1)
plt.title("Classification")
x0, x1 = clf.plot_decision_boundary(X_test)
blue_class, red_class = plot_test_data(X_test, y_test)
ax1 = plt.plot(x0, x1)
ax2 = plt.plot(blue_class[:,0], blue_class[:,1], 'bo')
ax3 = plt.plot(red_class[:,0], red_class[:,1], 'ro')

fig_err = plt.figure(2)
plt.title("Error")
iterations = np.arange(len(clf.errors))
errors = np.array(clf.errors).reshape(-1, 1)
sqrd_errors = np.square(clf.errors).reshape(-1, 1)
e = np.hstack([errors, sqrd_errors]).reshape(-1, 2)
ax4 = plt.plot(iterations, e)
plt.legend(["Error", "Squared Error"])

plt.show()

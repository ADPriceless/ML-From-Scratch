'''
Naive Bayes based on explanation at: 
https://www.machinelearningplus.com/predictive-modeling/how-naive-bayes-algorithm-works-with-example-and-full-code/

Bayes' Theorum:
p(y|x) = [p(x|y)*p(y)]/p(x)

This uses Bayes' Theorum for multiple classes (y) and features (x):
p(Y=k|X1, X2, ... Xn) = [p(X1, X2, ... Xn|Y=k)*p(Y=k)]/p(X1, X2, ... Xn)
'''

import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd

# Constants
NUM_BANANAS = 500
NUM_ORANGES = 300
NUM_OTHER = 200
TOTAL = NUM_BANANAS + NUM_ORANGES + NUM_OTHER

# Functions/Classes
def multiply_array_elements(arr):
    # Multipy all elements in array together
    result = 1.0
    for elem in arr:
        result *= elem
    return result

class NaiveBayes:
    def __init__(self):
        pass

    def calculate_prob_of_evidence(self, count_df, total):
        count_table = np.array(count_df)
        count_of_features = np.array(np.sum(count_table[c,:] for c in range(len(classes))))        
        prob_of_features = np.delete(count_of_features, np.s_[-1:], None)/total # delete 'Total' column
        prob_of_evidence = multiply_array_elements(prob_of_features)
        return prob_of_evidence

    def calculate_PoLoE(self, count_df):
        count_table = np.array(count_df, dtype=float)
        for class_ in count_table:
            class_ /= class_[-1:] # Find p(x_n | Y=c)
        prob_table = np.delete(count_table, np.s_[-1:], axis=1) # remove 'Total' column
        PoLoE = np.ones(prob_table.shape[1])

        # Calc p(X1|Y=c) * p(X2|Y=c) * ... * p(Xn|Y=c) for all classes, c 
        for i, class_ in enumerate(prob_table):
            for prob_of_feature_given_class in class_:
                PoLoE[i] *= prob_of_feature_given_class
        return PoLoE


    def predict(self, count_df, features, classes): # TODO: add Laplace correction
        # Calculate components of p(outcome|feature_evidence)
        total = np.sum(count_df['Total'])
        prior = np.array(count_df['Total'])/total   # p(Y=k) for all k
        prob_of_evidence = self.calculate_prob_of_evidence(count_df, total)
        prob_of_likelihood_of_evidence = self.calculate_PoLoE(count_df)
        # logging.info(prob_of_likelihood_of_evidence)
        # logging.info(prior)
        # logging.info(prob_of_evidence)
        
        # Calculate p(Y=k|X) for each k
        prob_of_outcome_given_evidence = \
            (prob_of_likelihood_of_evidence*prior)/prob_of_evidence

        # Return most likely fruit
        most_likely_outcome = np.max(prob_of_outcome_given_evidence)
        for i, p in enumerate(prob_of_outcome_given_evidence):
            if most_likely_outcome == p:
                return classes[i]

# Make data for all classes
fruit_df = pd.DataFrame(
    np.array([
    [400, 350, 450, NUM_BANANAS], 
    [0, 150, 300, NUM_ORANGES], 
    [100, 150, 50, NUM_OTHER]]))
features = ['Long', 'Sweet', 'Yellow', 'Total']
classes = ['Banana', 'Orange', 'Other']
fruit_df.rename({i:c for i, c in enumerate(classes)}, axis='index', inplace=True)
fruit_df.rename(columns={i:f for i, f in enumerate(features)}, inplace=True)
print(fruit_df)

clf = NaiveBayes()
print('\nMost likey fruit is', clf.predict(fruit_df, features, classes))

#!/usr/bin/env python

import csv
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import f_regression
from sklearn import linear_model


"""
Read the data from a CSV file and put it all in an array.
Assume a title line at the begining of the file.
@param path - the path to the file to read.
@return - an array of tuples of the form (array of features, array of target).
example of data: [
    ([1,2,3,4,5], [6,7]),
    ([2,1,4,4,6], [3,5]),
    ([4,2,3,5,5], [5,5])
    ...
]
"""
def readData(path, shuffle=False, test=False):
    data = []
    num_features = 0
    with open(path, 'rb') as datafile:
        reader = csv.reader(datafile)
        example = 0
        for row in reader:      # go through each line:

            if example == 0:    # first line assumed to be title line,
                num_features = len(row) # so get the number of features.
            
            if test and example > 10:
                break

            elif example > 0:   # for all other lines, grab the data.
                if len(row) != num_features: # check that nuber of features is correct
                    print "ERROR: number of features for this row is not %d" % num_features
                    print row
                    continue

                features = map(float,row[1:4]+row[6:]) # skip column 0, 4 and 5.
                target = map(float,row[4:6]) # targets are the 4th and 5th columns.
                
                assert len(features)+len(target)+1 == num_features
                
                data.append((features, target)) # add the tuple for this example.

            example=example+1 # increment example counter

    if shuffle:
        random.shuffle(data)
    return data


"""
@author: Nicolas A.G.

attributes:
- clf : linear regression estimator
- X : the whole X matrix
- Y : the whole Y matrix
- k : the number of partitions for X and Y.
- Xpartitions : array of the form [array of features #1, ..., array of features #k].
- Ypartitions : array of the form [array of targets #1, ..., array of targets #k].
- Ws : array of the best weight vectors.
- err: average of least-square error for all W's.
"""
class NicoRegression:
    
    """
    Constructor function.
    @param f - the file to open.
    @param lasso - decide if lasso regularization is done (default=False).
    @param ridge - decide if ridge regularization is done (default=False).
    @param k - the number of features to keep for k-best feature selection (default=not activated).
    @param rfe - decide if we do Recursive Feature Elimination (default=False).
    @param d - the dimension of linear regression (default=1)
    @param shuffle_data - decide if the data is shuffled (default=True).
    @param test - decide if we are testing, so return only 10 examples (default=False).
    """
    def __init__(self, f, lasso=False, ridge=False, k=-1, rfe=False, d=1, shuffle_data=True, test=False):
        data = readData(f, shuffle=shuffle_data, test=test)
        self.X = np.matrix([example[0] for example in data])
        self.Y = np.matrix([example[1] for example in data])

        if d > 1:
            self.X = PolynomialFeatures(degree=d).fit_transform(self.X)

        if k > 0 and k < 19:
            self.X = SelectKBest(f_regression, k=k).fit_transform(self.X, self.Y)

        alphas = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e+1]
        if ridge:
            print "RIDGE regularization"
            # generalized Cross-Validation over many different alphas
            self.clf = linear_model.RidgeCV(alphas=alphas)
        elif lasso:
            print "LASSO regularization"
            # generalized Cross-Validation over many different alphas
            self.clf = linear_model.MultiTaskLassoCV(alphas=alphas)
        else:
            print "no regularization"
            self.clf = linear_model.LinearRegression() # regular linear regression without regularization.

        if rfe:
            self.clf = RFECV(self.clf) # do recursive feature elimination with cross validation.



    """
    Apply PCA to the X matrix.
    @param m - the number of parameters we want to keep.
    """
    def doPCA(self, m):
        if m>0 and m<self.X.shape[1] and m<self.X.shape[0]:
            p = PCA(n_components=m)
            self.X = np.matrix(p.fit_transform(self.X))
            assert self.X.shape == (5875,m)
        else:
            print "ERROR: number of features has to be between 1 and %d" % min(self.X.shape)-1

    """
    Partitions the data into k subsets of size len(X)/k.
    @param k - the number of subsets we want.
    Xpartitions of the form [array of features #1, ..., array of features #k].
    Ypartitions of the form [array of targets #1, ..., array of targets #k].
    """
    def partition(self, k):
        print "partitioning..."
        self.k = k
        size = len(self.X) / int(k) # number of items in 1 subset.
        
        # let's go step by step with this one-liner:
        # we create an array: Xtrains = [...].
        # this array is made of subsets of X from i to i+size: self.X[i:i+size].
        # so we have: [ [data[i],data[i+1],...,data[i+size-1]], [...], ... ].
        # i is going to be an index from 0 to len(X), but by jumping over 'size' values,
        #  which leaves space for our 'size' data items (at i,i+1,...,i+size-1) in one subset.
        self.Xpartitions = [self.X[i:i+size] for i in range(0, len(self.X), size)]
        self.Ypartitions = [self.Y[i:i+size] for i in range(0, len(self.Y), size)]
        
        # because len(X) / k might not be an perfect integer, we may have fewer examples in the last subset.
        if len(self.X)%k != 0:
            print "WARNING: %d is not a multiple of %d. Skipping %d elements." % (len(self.X), k, len(self.X)%k)
            self.Xpartitions = self.Xpartitions[:-1]
            self.Ypartitions = self.Ypartitions[:-1]
        print "produced %d subsets of %d elements each." % (k, size)

        assert len(self.Xpartitions) == k
        assert len(self.Ypartitions) == k
        print "done partitioning."

    """
    W[i] = (Xt*X)^-1 * Xt * Y
    W[i] uses Xtrain = Xpartitions[:i][i+1:]
    W[i] uses Ytrain = Ypartitions[:i][i+1:]
    So W[i] is the best weight vector learned with all data EXCEPT partition i.
    err = average of | (W[i]^t * X) - Y |
    """
    def generateWs(self):
        print "generating Ws and Err..."
        #self.Ws = []
        self.err = [0.0, 0.0] # 2 errors for the two targets.

        for i in range(self.k):
            Xtrain = self.Xpartitions[:i]+self.Xpartitions[i+1:]
            Xtrain = reduce(lambda x,y: np.concatenate((x,y)), Xtrain)
            Xtest = self.Xpartitions[i]
            
            Ytrain = self.Ypartitions[:i]+self.Ypartitions[i+1:]
            Ytrain = reduce(lambda x,y: np.concatenate((x,y)), Ytrain)
            Ytest = self.Ypartitions[i]

            self.clf.fit(Xtrain, Ytrain) # learn the model.
            if self.clf.alpha_:
                print "  %i - alpha: %e" %(i, self.clf.alpha_)
            if self.clf.n_features_:
                print "  Optimal number of features : %d" % self.clf.n_features_

            #W = np.matrix(self.clf.coef_).getT() # get the best Weights.
            #self.Ws.append(W) # append to Ws.

            mse = np.power(self.clf.predict(Xtest)-Ytest, 2).mean(0) # get the mean squarred error.
            self.err += mse # add mse to the overall error.

        self.err = map(lambda x:x/self.k, self.err)
        print "done."

nico = NicoRegression("./data.csv", lasso=True) # 5875 elements, try ridge/lasso/k(1-18)/rfe

print "X shape:", nico.X.shape
print "Y shape:", nico.Y.shape
nico.partition(5875) #try with 5, 25, 47, 125, 235, 1175, 5875

nico.generateWs()
print nico.err



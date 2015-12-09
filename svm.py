#!/usr/bin/env python

import csv
import random
import numpy as np
from datetime import datetime
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import f_regression


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
def readData(path, shuffle=True, test=False):
    data = []
    num_features = 0
    with open(path, 'rb') as datafile:
        reader = csv.reader(datafile)
        example = 0
        for row in reader:      # go through each line:

            if example == 0:    # first line assumed to be title line,
                num_features = len(row) # so get the number of features.
            
            if test and example > 100:
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
attributes:
- clf : linear regression estimator
- X : the whole X matrix
- Y : the whole Y matrix
- k : the number of partitions for X and Y.
- Xpartitions : array of the form [array of features #1, ..., array of features #k].
- YmotorPartitions : array of the form [motor target #1, ..., motor target #k].
- YtotalPartitions : array of the form [total target #1, ..., total target #k].
- Ws : array of the best weight vectors.
- err: average of least-square error for all W's.
"""
class NicoSVM:
    
    """
    Constructor function.
    @param f - the file to open.
    @param C - tradeoff misclassification of training examples against simplicity of the decision surface.
    @param gamma - gamma of the RBF kernel (how much influence a single training example has).
    @param d - the dimension of features (default=1)
    @param k - the number of features to keep for k-best feature selection (default=not activated).
    @param shuffle_data - decide if the data is shuffled (default=True).
    @param test - decide if we are testing, so return only 10 examples (default=False).
    """
    def __init__(self, f, C, gamma, d=1, k=-1, shuffle_data=True, test=False):
        data = readData(f, shuffle=shuffle_data, test=test)
        self.X = np.matrix([example[0] for example in data])
        self.Y = np.matrix([example[1] for example in data])
        self.Y1 = self.Y[:,0]
        self.Y2 = self.Y[:,1]
        #print "X shape:", self.X.shape
        #print "Y shape:", self.Y.shape
        #print "Y1 shape:", self.Y1.shape
        #print "Y2 shape:", self.Y2.shape
        
        self.clf = svm.SVR(kernel='rbf', C=C, gamma=gamma, cache_size=1000)

        if d > 1:
            print "polynomial features of degree %d" % d
            self.X = PolynomialFeatures(degree=d).fit_transform(self.X)

        if k > 1:
            print "select %d best features" % k
            self.X = SelectKBest(f_regression, k=k).fit_transform(self.X, np.squeeze(np.asarray(self.Y2)))

        #print "X shape:", self.X.shape
        #print "Y shape:", self.Y.shape
        #print "Y1 shape:", self.Y1.shape
        #print "Y2 shape:", self.Y2.shape



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
        # we create an array: Xpartitions = [...].
        # this array is made of subsets of X from i to i+size: self.X[i:i+size].
        # so we have: [ [data[i],data[i+1],...,data[i+size-1]], [...], ... ].
        # i is going to be an index from 0 to len(X), but by jumping over 'size' values,
        #  which leaves space for our 'size' data items (at i,i+1,...,i+size-1) in one subset.
        self.Xpartitions = [self.X[i:i+size] for i in range(0, len(self.X), size)]
        self.Ypartitions = [self.Y[i:i+size] for i in range(0, len(self.Y), size)]
        self.Y1Partitions = [self.Y1[i:i+size] for i in range(0, len(self.Y1), size)]
        self.Y2Partitions = [self.Y2[i:i+size] for i in range(0, len(self.Y2), size)]
        
        # because len(X) / k might not be an perfect integer, we may have fewer examples in the last subset.
        if len(self.X)%k != 0:
            print "WARNING: %d is not a multiple of %d. Skipping %d elements." % (len(self.X), k, len(self.X)%k)
            self.Xpartitions = self.Xpartitions[:-1]
            self.Ypartitions = self.Ypartitions[:-1]
            self.Y1Partitions = self.Y1Partitions[:-1]
            self.Y2Partitions = self.Y2Partitions[:-1]
        print "produced %d subsets of %d elements each." % (k, size)

        assert len(self.Xpartitions) == k
        assert len(self.Ypartitions) == k
        assert len(self.Y1Partitions) == k
        assert len(self.Y2Partitions) == k
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
        self.err = [0.0, 0.0] # 2 errors for the two targets.

        for i in range(self.k):
            Xtrain = self.Xpartitions[:i]+self.Xpartitions[i+1:]
            Xtrain = reduce(lambda x,y: np.concatenate((x,y)), Xtrain)
            Xtest = self.Xpartitions[i]
            
            Ytrain = self.Ypartitions[:i]+self.Ypartitions[i+1:]
            Ytrain = reduce(lambda x,y: np.concatenate((x,y)), Ytrain)
            Ytest = self.Ypartitions[i]

            Ytrain_motor = Ytrain[:,0]
            Ytest_motor = Ytest[:,0]
            # Fit the model for MOTOR target.
            self.clf.fit(Xtrain, np.squeeze(np.asarray(Ytrain_motor)))
            print "  %i - motor" % i
            #print "  Optimal number of features : %d" % self.clf.n_features_
            testing_motorPredictions = np.matrix(self.clf.predict(Xtest))
            mse = np.power(testing_motorPredictions-Ytest_motor.T, 2).mean(1)
            self.err[0] += mse

            Ytrain_total = Ytrain[:,1]
            Ytest_total = Ytest[:,1]
            # Add prediction of motor to Xtrain and Xtest to better predict total.
            training_motorPredictions = np.matrix(self.clf.predict(Xtrain))
            Xtrain = np.append(Xtrain, training_motorPredictions.T, axis=1)
            Xtest = np.append(Xtest, testing_motorPredictions.T, axis=1)
            # Fit the model for TOTAL target.
            self.clf.fit(Xtrain, np.squeeze(np.asarray(Ytrain_total)))
            print "  %i - total" % i
            #print "  Optimal number of features : %d" % self.clf.n_features_
            mse = np.power(self.clf.predict(Xtest)-Ytest_total.T, 2).mean(1)
            self.err[1] += mse

        self.err = map(lambda x:x/self.k, self.err)
        print "done."

'''USE RBF kernel
cf: http://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel
'''
#C_range = [1e2, 1e3, 1e4, 1e5]
#gamma_range = [1e-6, 1e-5]

#for C in C_range:
#    for gamma in gamma_range:
C = 1000.0
gamma = 0.00001
d = 2
k_max = 210
for k in range(54,k_max+1):
    print "RBF kernel"
    print "C: %e" % C
    print "gamma: %e" % gamma
    print "#of features: %d" % k

    start = datetime.now()

    nico = NicoSVM("./data.csv", C=C, gamma=gamma, d=d, k=k) # 5875 elements
    nico.partition(5) #try with 5, 25, 47, 125, 235, 1175, 5875
    nico.generateWs()
    print nico.err

    with open('out.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([
            "k=%d" % k,
            float(np.asarray(nico.err[0])[0][0]),
            float(np.asarray(nico.err[1])[0][0])
        ])

    print datetime.now() - start

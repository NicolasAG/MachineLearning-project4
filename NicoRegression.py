#!/usr/bin/env python

import csv
import numpy as np
import random

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
- X : the whole X matrix
- Y : the whole Y matrix
- k : the number of partitions for X and Y.
- Xpartitions : array of the form [array of features #1, ..., array of features #k].
- Ypartitions : array of the form [array of targets #1, ..., array of targets #k].
- W : array of the best weight vectors.
- err: average of least-square error for all W's.
"""
class NicoRegression:

    def __init__(self, file, shuffle=False, test=False):
        data = readData(file, shuffle=shuffle, test=test)
        self.X = np.matrix([example[0] for example in data])
        self.Y = np.matrix([example[1] for example in data])

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
        self.W = []
        self.err = [0.0, 0.0]

        for i in range(self.k):
            Xtrain = self.Xpartitions[:i]+self.Xpartitions[i+1:]
            Xtrain = reduce(lambda x,y: np.concatenate((x,y)), Xtrain)
            Xtest = self.Xpartitions[i]
            
            Ytrain = self.Ypartitions[:i]+self.Ypartitions[i+1:]
            Ytrain = reduce(lambda x,y: np.concatenate((x,y)), Ytrain)
            Ytest = self.Ypartitions[i]

            W = (Xtrain.getT()*Xtrain).getI()*Xtrain.getT()*Ytrain
            self.W.append(W)
            
            err_ = np.power((Xtest*W - Ytest), 2)
            self.err += err_.mean(0)

        self.err = map(lambda x:x/self.k, self.err)
        print "done."

nico = NicoRegression("./data.csv", shuffle=True, test=False) # 5875 elements
print nico.X.shape
print nico.Y.shape

nico.partition(5) #try with 5, 25, 47, 125, 235, 1175, 5875
nico.generateWs()

print nico.err

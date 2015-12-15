
import csv
import numpy as np
from sklearn import preprocessing
from sknn.mlp import Regressor , Layer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
"""
Read the data from a CSV file and put it all in an array.
Assume a title line at the begining of the file.
@param path - the path to the file to read.
@return - an array of tuples of the form (array of features, array of target).
example of data:
  [
    ([1,2,3,4,5], [6,7]),
    ([2,1,4,4,6], [3,5]),
    ([4,2,3,5,5], [5,5])
    ...
                          ]
"""
def readData(path):
    data = []
    num_features = 0
    with open(path, 'rb') as datafile:
        reader = csv.reader(datafile)
        example = 0
        for row in reader:      # go through each line:

            if example == 0:    # first line assumed to be title line,
                num_features = len(row) # so get the number of features.

            elif example > 0:   # for all other lines, grab the data.
                if len(row) != num_features: # check that nuber of features is correct
                    print "ERROR: number of features for this row is not %d" % num_features
                    print row
                    continue

                features = map(float,row[6:])#row[1:4]+row[6:]) # skip column 0, 4 and 5.
                target = map(float,row[4:6]) # targets are the 4th and 5th columns.


                data.append((features, target)) # add the tuple for this example.
            example=example+1 # increment example counter


    return data


def main():

    print "Loading Data..."
    data = readData("./data.csv") #, shuffle=shuffle_data, test=test)
    # Ymotor = np.squeeze(np.asarray([example[1][0] for example in data[0]]))
    # Ytotal= np.squeeze(np.asarray([example[1][1] for example in data[0]]))

    # X = SelectKBest(f_regression, k=9).fit_transform(X, target[:,1])
    # print  X.shape

    # print Ytotal
    # print Ymotor.shape
    # print Ytotal, "," , Ymotor
    # print X

    # degrees = [1, 2, 3]
    # for i in range(len(degrees)):


    Error= []
    MSE=[]
    Avg_score= []
    record = []

    # X = SelectKBest(f_regression, k=9).fit_transform(X, target[:,1])
    # # print  X.shape
    #
    # polynomial_features = PolynomialFeatures(degree=1,include_bias=True)
    # X= polynomial_features.fit_transform(X)
    # # X= np.c_[np.ones(len(X)),X] #concatente 2 columns vertically ;)
    # # print len(X[1])
    # # print  X[0,:]



    for layer in [[400,5]]:#,[400,5],[400,6],[400,7],[450,6],[450,7],[485,5],[485,6],[485,7],[500,6],[600,300],[800,6],[800,500],[1000,6],[1000,100],[1000,500]]:

        for lr in [0.0009]:
        # for layer in [[6],[10],[15],[20],[25],[30],[35],[40],[50],[60],[75],[100],[400],[600]]:

            for alpha in [0.9]:#[0.01, 0.1, 0.7, 0.8 , 0.9]:

                # for k in range(6,8):   #k=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                # for n_folds in range(10,11):

                X = np.squeeze(np.asarray([example[0] for example in data])) #Total examples of 5875
                target= np.squeeze(np.asarray([example[1] for example in data]))

                polynomial_features = PolynomialFeatures(degree=2, include_bias=True)
                X= polynomial_features.fit_transform(X)
                # print len(X[1])
                # print  X[0,:]

                X = SelectKBest(f_regression, k=6).fit_transform(X, target[:,1])
                # print  X.shape



                n_folds =5

                kf = KFold(len(X[0]), n_folds)
                # print len(kf)
                # print(kf)
                for train_index, test_index in kf:
                    # print("TRAIN:", train_index, "TEST:", test_index)
                    train_X, valid_X = X[train_index], X[test_index]
                    train_target, valid_target = target[train_index], target[test_index]



                    # print 'Standardizing...'
                    scaler = preprocessing.StandardScaler().fit(train_X)  #fit only on training data (Compute the mean and std to be used for later scaling)
                    train_set = scaler.transform(train_X)  #Perform standardization by centering and scaling
                    valid_set = scaler.transform(valid_X) # apply same transformation to test data
                    # print train_X[:4]
                    # print train_set[:4]



                    #
                    # nn = Regressor(
                    # layers=[
                    #     # Layer("Sigmoid", units=1000),
                    #     Layer("Sigmoid", units=500),
                    #     Layer("Linear", units=6),
                    #     Layer("Linear", units=2)],
                    # learning_rule='sgd',
                    # regularize='L2',
                    # weight_decay=0.7,
                    # learning_rate=0.0009,
                    # batch_size=30,
                    # n_iter=10,
                    # loss_type='mse',)



                    nn = Regressor(
                        layers=[
                            Layer("Sigmoid", units=layer[0]),
                            # Layer("Sigmoid", units=600),
                            Layer("Linear", units=layer[1]),
                            Layer("Linear", units=2)],
                        learning_rule='sgd',
                        regularize='L2',
                        weight_decay= alpha,
                        learning_rate=lr,
                        batch_size=30,
                        n_iter=10,
                        loss_type='mse',)

                    nn.fit(train_set, train_target)
                    y_example = np.squeeze(np.asarray(nn.predict(valid_set)))
                    # print valid_target, y_example
                    Error.append(np.absolute(valid_target - y_example))
                    MSE.append(np.power(Error[-1], 2).mean(0))
                    # MSE= np.mean(MSE, axis=0)
                    # print 'Fold:',f, np.matrix(MSE).mean(0)
                y1= np.array([example[0] for example in MSE]).tolist()
                y2= np.array([example[1] for example in MSE]).tolist()

                Avg_score= np.matrix(zip(y1,y2)).min(0)

                #
                # record.append([lr,layer_size_list,err_list[best_fold]])
                # record.append([k,Avg_score])
                # print record
                print  " layer:", layer,","," lr:",lr,",","alpha:",alpha,",", np.around(Avg_score,decimals=3)





if __name__=="__main__":
    main()
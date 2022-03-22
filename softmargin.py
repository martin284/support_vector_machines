import numpy as np
from sklearn import datasets
import math

def softmargin(data0, data1, supportvec):
    # count missaligned samples
    if data0[supportvec[0]] < data1[supportvec[1]]:
        data0_miss = np.sum(data0 > data0[supportvec[0]])
        data1_miss = np.sum(data1 < data1[supportvec[1]])
    else:
        data0_miss = np.sum(data0 < data0[supportvec[0]])
        data1_miss = np.sum(data1 > data1[supportvec[1]])
    # calculate thresh
    thresh = np.mean([data0[supportvec[0]], data1[supportvec[1]]])
    # calculate negative minimal max margin
    margin = np.minimum(np.min(np.abs(data0-thresh)),
    np.min(np.abs(data1-thresh)))
    # return the score
    return data0_miss + data1_miss - margin

def calculate_support_vectors(data0, data1):
    # build matrix for saving the scores
    scores = np.zeros([len(data0), len(data1)])
    # calculate score for every combination of support vectors
    for i in range(len(data0)):
        for j in range(len(data1)):
            scores[i, j] = softmargin(data0, data1, [i, j])
    # find minimal (best) score and corresponding support vectors
    min_score = np.min(scores)
    support_vectors = np.argwhere(scores == np.min(scores))[0]
    # return results
    return support_vectors, min_score

def test_support_vectos(data0, data1, supportvec):
    if data0[supportvec[0]] < data1[supportvec[1]]:
        data0_miss = np.sum(data0 > data0[supportvec[0]])
        data1_miss = np.sum(data1 < data1[supportvec[1]])
    else:
        data0_miss = np.sum(data0 < data0[supportvec[0]])
        data1_miss = np.sum(data1 > data1[supportvec[1]])
    return data0_miss + data1_miss

if __name__ == '__main__':
    # load iris data
    iris = datasets.load_iris()
    # only use information about petal width
    data = iris.data[:, 3]
    # assign data to training data or test data randomly
    idx_use = np.random.permutation(iris.target.size)
    # leave out 30% of data for testing
    training_data = data[idx_use[:100]]
    test_data = data[idx_use[100:]]
    # get labels
    training_labels = iris.target[idx_use[:100]]
    test_labels = iris.target[idx_use[100:]]
    # treat labels from all other classes as one class
    training_labels[training_labels > 0] = 1
    test_labels[test_labels > 0] = 1
    # split data in two data sets
    data0 = training_data[training_labels == 0]
    data1 = training_data[training_labels == 1]
    # compute optimal support vectors
    supportvec, score = calculate_support_vectors(data0, data1)
    # test support vectors
    n_missaligned_data = test_support_vectos(data0, data1, supportvec)
    # print results
    print('Support Vectors:', supportvec)
    print('Score:', score)
    print('Number of missaligned data points:', n_missaligned_data)

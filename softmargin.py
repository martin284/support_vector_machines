import numpy as np

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
    margin = np.minimum(np.min(np.abs(data0-thresh)), np.min(np.abs(data1-thresh)))
    # return the score
    return data0_miss + data1_miss - margin

if __name__ == '__main__':
    data0 = np.array([1, 2, 3, 11])
    data1 = np.array([10, 14, 18])
    supportvec = [2, 0]
    score = softmargin(data0, data1, supportvec)
    print('score: ' + str(score))

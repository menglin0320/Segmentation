# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numpy.core.umath_tests import matrix_multiply
from numpy.linalg import svd
from sklearn.decomposition import PCA

#f = np.array([[[0, 1, 2], [10, 11, 12]], [[3, 4, 5], [6, 7, 8]]], dtype = np.float)
def procrustes2d(f):
    # zero mean normalize input aray
    mean = np.mean(np.mean(f, axis = 0), axis = 1).reshape(2, 1)
    print(mean)
    print(np.shape(mean))
#    f = (f.T - mean).T
    f = f - mean
    print(f)
    
    # pick the first element as pivot
    # scale pivot element to 1
    pivot = f[0]
#    print(pivot)
#    print(np.shape(pivot))
    scale = np.sqrt(np.sum(np.power(pivot-mean, 2))/np.shape(pivot)[1])
#    print(scale)
    pivot = (pivot-mean)/scale
    print(pivot)
#    scale1 = np.sqrt(np.sum(np.power(pivot, 2))/np.shape(pivot)[1])
#    print(scale1)    
    
    diff = np.inf
    prevDiff = 0    
    while(diff - prevDiff > 0.0001):  
        # do the rotation
        f[0] = pivot
        bot = np.einsum('kij, ij...->ki', f, pivot)
        print(bot)
        bot1 = np.sum(bot, axis = 1)
        print(bot1)
        pivot = np.roll(pivot, 1, axis = 0)
        pivot[1] = -pivot[1]
        print(pivot)
        top = np.einsum('kij, ij...->ki', f, pivot)
        print(top)
        top1 = np.sum(top, axis = 1)
        print(top1)
        theta = np.arctan2(bot1, top1)
        print(theta)

        sine = np.sin(theta)
        print('sine:')
        print(sine)
        cosine = np.cos(theta)
        print('cosine:')
        print(cosine)
        transformMatrix = np.zeros((np.shape(f)[0] , 4))

        #build transfor matrix
        transformMatrix[:, 0] = cosine
        transformMatrix[:, 1] = -sine
        transformMatrix[:, 2] = sine
        transformMatrix[:, 3] = cosine
        transformMatrix = transformMatrix.reshape(np.shape(f)[0], 2, 2)
        print(transformMatrix)

        update = matrix_multiply(transformMatrix, f)
        print(update)
        print(np.shape(update))

        diff = np.sqrt(np.sum(np.square(f-update)))
        print(diff)
    
#procrustes2d(f)

f = np.array([[[0, 1, 2], [10, 11, 12], [0, 1, 2]], [[3, 4, 5], [6, 7, 8], [0, 1, 2]]], dtype = np.float)
def procrustes3d(f):
    # zero mean normalize input aray
    mean = np.mean(np.mean(f, axis = 0), axis = 1).reshape(3, 1)
    print(mean)
    print(np.shape(mean))
#    f = (f.T - mean).T
    f = f - mean
    print(f)
    
    # pick the first element as pivot
    # scale pivot element to 1
    pivot = f[0]
#    print(pivot)
#    print(np.shape(pivot))
    scale = np.sqrt(np.sum(np.power(pivot-mean, 2))/np.shape(pivot)[1])
#    print(scale)
    pivot = (pivot-mean)/scale
    print(pivot)
#    scale1 = np.sqrt(np.sum(np.power(pivot, 2))/np.shape(pivot)[1])
#    print(scale1)    
    
    diff = np.inf
    prevDiff = 0 
    while(diff - prevDiff > 0.0001):
        prevf = f        
        
        # do the rotation
        f[0] = pivot
        for i in range(1, np.shape(f)[0]):
            u, s, v = svd(f[i].T * pivot)
            transformMatrix = v * u.T
            f[i] = matrix_multiply(transformMatrix, f[i])
            print(f[i])
        
        diff = np.sqrt(np.sum(np.square(f - prevf)))
        print(diff)

#procrustes3d(f)

#pca
def pca(matrixAfterPro, metric, isWhiten):
    if(type(metric) != float):
        raise Exception("metric type error!")
        
    shape = np.shape(matrixAfterPro)
    matrixAfterPro = np.reshape(matrixAfterPro, (shape[0], shape[1]*shape[2]))
    
    pca = PCA(n_components = metric, whiten = isWhiten)
    matrixAfterPCA = pca.fit_transform(matrixAfterPro)
    
    return matrixAfterPCA

print(pca(f, np.float(0.98), False))

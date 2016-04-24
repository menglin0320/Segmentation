# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numpy.core.umath_tests import matrix_multiply
from numpy.linalg import svd
from sklearn.decomposition import PCA
import random

f = np.array([[[0, 1, 2], [10, 11, 12]], [[3, 4, 5], [6, 7, 8]]], dtype = np.float)
def procrustes2d(f):
    numberOfShapes = np.shape(f)[0]  
    diff = np.inf   
    timesToConverge = 0
    while(abs(diff) > 0.01):
        timesToConverge += 1
    
        # zero mean normalize input aray
        mean = np.mean(np.mean(f, axis = 0), axis = 1).reshape(2, 1)
        print(mean)
        print(np.shape(mean))
        #f = (f.T - mean).T
        f = f - mean
        print(f)
    
        #pick radom element as pivot
        #scale pivot element to 1
        index = random.randint(0, numberOfShapes - 1)
        pivot = f[index]
#        print(pivot)
#        print(np.shape(pivot))
        scale = np.sqrt(np.sum(np.power(pivot-mean, 2))/np.shape(pivot)[1])
        print("scale", scale)
        pivot = (pivot-mean)/scale
        print(pivot)
        f[index] = pivot
#        scale1 = np.sqrt(np.sum(np.power(pivot, 2))/np.shape(pivot)[1])
#        print(scale1)
    
        # do the rotation
        bot = np.einsum('kij, ij...->ki', f, pivot)
#        print(bot)
        bot1 = np.sum(bot, axis = 1)
#        print(bot1)
        pivot = np.roll(pivot, 1, axis = 0)
        pivot[1] = -pivot[1]
#        print(pivot)
        top = np.einsum('kij, ij...->ki', f, pivot)
#        print(top)
        top1 = np.sum(top, axis = 1)
#        print(top1)
        theta = np.arctan2(top1, bot1)
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
        print("before reshape", transformMatrix)
        transformMatrix = transformMatrix.reshape(np.shape(f)[0], 2, 2)
        print("after reshape", transformMatrix)

        update = matrix_multiply(transformMatrix, f)
        print(update[index] == pivot)
        print("update:", update)

        diff = np.sqrt(np.sum(np.square(f-update)))
        f = update
        print(diff)
    
    print(timesToConverge)
procrustes2d(f)

f = np.array([[[0, 1, 2], [10, 11, 12], [0, 1, 2]], [[3, 4, 5], [6, 7, 8], [0, 1, 2]]], dtype = np.float)
def procrustes3d(f):
    numberOfShapes = np.shape(f)[0]    
    timesToConverge = 0
    diff = np.inf    
    while(diff > 1):
        prevf = np.copy(f)
        
        # zero mean normalize input aray
        mean = np.mean(np.mean(f, axis = 0), axis = 1).reshape(3, 1)
#        print(mean)
#        print(np.shape(mean))
#        f = (f.T - mean).T
        f = f - mean

        #pick radom element as pivot
        #scale pivot element to 1
        index = random.randint(0, numberOfShapes - 1)
        pivot = f[index]
        #print(pivot)
        #print(np.shape(pivot))
        scale = np.sqrt(np.sum(np.power(pivot-mean, 2))/np.shape(pivot)[1])
        #print(scale)
        pivot = (pivot-mean)/scale
        print(pivot)
        #scale1 = np.sqrt(np.sum(np.power(pivot, 2))/np.shape(pivot)[1])
        #print(scale1)    
        
        timesToConverge += 1  
        f[index] = pivot
        # do the rotation
        for i in range(0, np.shape(f)[0]):
            if(i == index):
                continue
#            print(f[i])
            u, s, v = svd(f[i].T * pivot)
            transformMatrix = v * u.T
            f[i] = matrix_multiply(transformMatrix, f[i])
            print(f[i])
        
        diff = np.sqrt(np.sum(np.square(f - prevf)))
        print(diff)
    
    print(timesToConverge)
    return f

#print(procrustes3d(f))

#pca
def pca(matrixAfterPro, metric, isWhiten):
    if(type(metric) != float):
        raise Exception("metric type error!")
        
    shape = np.shape(matrixAfterPro)
    matrixAfterPro = np.reshape(matrixAfterPro, (shape[0], shape[1]*shape[2]))
    
    pca = PCA(n_components = metric, whiten = isWhiten)
    matrixAfterPCA = pca.fit_transform(matrixAfterPro)
    
    return matrixAfterPCA

#print(pca(f, np.float(0.98), False))

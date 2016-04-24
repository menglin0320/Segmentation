# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 22:34:45 2016

@author: Kangyan Zhou
"""
import os  
import redis
import re
import numpy as np
from PIL import Image
from matplotlib import pyplot

fold1Path = "./medical_data/fold1/"
fold2Path = "./medical_data/fold2/"
patternBracket1 = '\[(.+)\]'
patternBracket2 = '\{(.+)\}'

def initializeReids():
     r = redis.Redis(host='localhost', port=6379, db=0) 
     
     return r

def loadData_Landmarks(isTraining):
    if(isTraining):
        currPath = fold1Path + 'landmarks/'
    else:
        currPath = fold2Path + 'landmarks/'
        
    allFileFeatures = {}
    for fn in os.listdir(currPath):
        oneFileInfo = []
        newFeature = []
        points = []
        with open(currPath + fn, 'r') as file:
            for line in file:
                if(line.startswith(";")):
                    continue
                
                if(line.startswith("{")):
                    points = []
                    newFeature = []
                    
                if(line.startswith("}")):
                    newFeature.append(points)
                    oneFileInfo.append(newFeature)
                    
                #getting label info
                m = re.search(patternBracket1, line)
                if(m):
                    info = m.group(1)
                    newFeature.append(info.split('=')[1])
                
                # get point info
                m = re.search(patternBracket2, line)
                if(m):
                    info = m.group(1)
                    pair = info.split(',')
                    pair[0] = float(pair[0])
                    pair[1] = float(pair[1])
                    points.append(pair)
                
        allFileFeatures[fn] = oneFileInfo
    
    return allFileFeatures

def loadData_Points(isTraining):
    if(isTraining):
        currPath = fold1Path + 'points/'
    else:
        currPath = fold2Path + 'points/'
        
    allFileFeatures = {}
    for fn in os.listdir(currPath):
        oneFileInfo = []
        newFeature = []
        points = []
        with open(currPath + fn, 'r') as file:
            for line in file:
                if(line.startswith(";")):
                    continue
                
                if(line.startswith("{")):
                    points = []
                    newFeature = []
                    
                if(line.startswith("}")):
                    newFeature.append(points)
                    oneFileInfo.append(newFeature)
                    
                #getting label info
                m = re.search(patternBracket1, line)
                if(m):
                    info = m.group(1)
                    newFeature.append(info.split('=')[1])
                
                # get point info
                m = re.search(patternBracket2, line)
                if(m):
                    info = m.group(1)
                    pair = info.split(',')
                    pair[0] = float(pair[0])
                    pair[1] = float(pair[1])
                    points.append(pair)
                
        allFileFeatures[fn] = oneFileInfo
    
    return allFileFeatures

# whats the format of images?
def loadData_Images(isTraining):
    if(isTraining):
        currPath = fold1Path + 'masks/'
    else:
        currPath = fold2Path + 'masks/'
    
    images = {}
    
    for dirName in os.listdir(currPath):
        currCategoryPath = currPath + dirName + '/'
        perCategory = {}
        for fn in os.listdir(currCategoryPath):
            img = Image.open(currCategoryPath + fn)
            arr = np.array(img)
            perCategory[fn] = arr
        
        images[dirName] = perCategory
        
    return images

temp1 = loadData_Landmarks(True)
temp2 = loadData_Points(True)
temp3 = loadData_Images(True)

# reference: http://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
def read_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def checkPlot(): 
    image = read_pgm("0000_02176.pgm", byteorder='<')
    image = read_pgm("0000_0_0_0_15_0_1.pgm", byteorder='<')
    pyplot.imshow(image, pyplot.cm.gray)
    pyplot.show()

checkPlot()
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold # import KFold

from sklearn.model_selection import train_test_split

csv_dataset = pd.read_csv("dataset/IRIS.csv", delimiter=',', header=0)
dataset = np.array(csv_dataset)                      #konversi dataset csv menjadi array
dataset = dataset.astype(float)                      #konversi dataset menjadi tipe float
n_dataset = len(dataset[:,0])                        #menghitung banyaknya dataset
n_feature = len(dataset[0,:]) - 1                    #membaca jumlah feature 


n_epoch_ = 10                                                     #Jumlah Epoh
_alfa_ = 0.1                                                    #learning Rate
# prepare cross validation
CFold = 5
data = dataset               
dataFiture = data[:,:n_feature]
dataLabel  = data[:,n_feature]
skf = StratifiedKFold(n_splits=CFold)
skf.get_n_splits(dataFiture, dataLabel)

dtTrain = []
dtTest = []
for train_index, test_index in skf.split(dataFiture, dataLabel):
	dtTrain.append(data[train_index])
	dtTest.append(data[test_index])

dtTrain = np.array(dtTrain)
dtTest = np.array(dtTest)

for a in range(0, CFold):
	np.random.shuffle(dtTrain[a])	
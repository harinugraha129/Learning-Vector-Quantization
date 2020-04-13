import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from training import weight,n_dataInitial,bobot, n_dataTraining, n_epoch, alfa_
from dataset import dtTest, dtTrain, n_feature, CFold
from distance import euclidean_D, manhattan_D, minkowski_D, cosine_D, canberra_D, chebishev_D


avg = 0
# Cross Fold Bergilir
for a in range(0, CFold):   
    accuracy = 0.0
    y_true = []
    y_pred = []
    
    # Menghitunga Banyak Data
    n_dataTesting	= len(dtTest[a])

    for i in range(0, n_dataTesting):
        data =[]
        label=[]
        #Menghitung Jarak Data
        for j in range(0, n_dataInitial ):

            x = dtTest[a][i]
            y = bobot[a][j]

            data.append(minkowski_D(x,y))
            label.append(bobot[a][j][n_feature])

            dataJarak = {'jarak' : pd.Series(data),
                            'label' : pd.Series(label)} 
        # end for
        df = pd.DataFrame(dataJarak, columns = ['jarak', 'label'])
        df = df.sort_values(by=['jarak'])

        labelTemp = df.iloc[0,1]
        labelTemp = int(labelTemp)

        y_true.append(dtTest[a][i][n_feature])
        y_pred.append(labelTemp)

        

    # Mendapatkan Akurasi Dan Recall
    # recall = recall_score(y_true, y_pred, average='micro')
    acc = accuracy_score(y_true, y_pred)
    print("Akurasi : " + str(acc*100)+" %")
    avg= avg + acc

print("Rata rata Akurasi : " + str(avg*100/CFold)+" %")
print("Jumlah Data Training : " +str(n_dataTraining))
print("Jumlah Data Testing : " +str(n_dataTesting))
print("Jumlah Fiture : " + str(n_feature))
print("Learning Rate : " + str(alfa_))
print("Jumlah Epoh : " + str(n_epoch))
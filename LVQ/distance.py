import math
from dataset import n_feature


def euclidean_D(x,y):
    distance = 0.0
    for i in range (0,n_feature):
        distance += (x[i]-y[i])**2
    return math.sqrt(distance)

def manhattan_D(x,y):
    distance = 0.0
    for i in range (0,n_feature):
        distance += math.fabs(x[i]-y[i])
    return distance

def minkowski_D(x,y):
    distance = 0.0
    for i in range (0,n_feature):
        distance += (math.fabs(x[i]-y[i]))**3
    return (distance)**1/3

def canberra_D(x,y):
    distance = 0.0
    distance2 = 0.0
    for i in range (0,n_feature):
        distance += (math.fabs(x[i]-y[i]))
        distance2 += (math.fabs(x[i]) + math.fabs(y[i]))
    return (distance/distance2)

def chebishev_D(x,y):
    distance = []
    for i in range (0,n_feature):
        distance.append(math.fabs(x[i]-y[i]))  
    return max(distance)

def cosine_D(x,y):
    distance = 0.0
    distance2 = 0.0
    distance3 = 0.0
    for i in range (0,n_feature):
        distance += (x[i]*y[i])
        distance3 += (x[i])**2
        distance2 += (y[i])**2
    
    distance2 = math.sqrt(distance2)
    distance3 = math.sqrt(distance3)
    distance = (distance/(distance2*distance3))
    return 1-distance
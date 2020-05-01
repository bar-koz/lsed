#!/usr/bin/ev python

import numpy as np
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

#Wczytywanie danych i tworzenie clustermap
columns = ("warm-blooded", "can fly", "vertebrate", "endangered", "live in groups", "have hair")
all_data = pd.read_csv("animals.csv", header=0, index_col=0)
all_data.columns=columns
all_data = all_data.fillna(0)
g = sns.clustermap(all_data, cmap="mako", vmin=0., vmax=2.)
plt.savefig('clustermap.png')

#Funkcja do tworzenia kombinacji
def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

#Wczytywanie danych zbioru iris
data = load_iris()
data_target = data.target
data = data.data
#Tworzenie kombinacji parametrow
data_par = (0,1,2,3)
com = combinations(data_par, 2)
#Kombinacjie 2 parametrow
for i in com:
	d = np.concatenate((data[:,i[0]:i[0]+1], data[:,i[1]:i[1]+1]),axis=1)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(d, data_target)
	print "Parametry: " + str(i[0]) + ", " + str(i[1])
	print np.mean(cross_val_score(kmeans, d, data_target, cv=5))
#Kombinacje 3 parametrow
com = combinations(data_par, 3)
for i in com:
	d = np.concatenate((data[:,i[0]:i[0]+1], data[:,i[1]:i[1]+1]),axis=1)
	d = np.concatenate((d, data[:,i[2]:i[2]+1]),axis=1)
	kmeans = KMeans(n_clusters=3, random_state=0).fit(d, data_target)
	print "Parametry: " + str(i[0]) + ", " + str(i[1]) + ", " + str(i[2])
	print np.mean(cross_val_score(kmeans, d, data_target, cv=5))
#4 parametry
kmeans = KMeans(n_clusters=4, random_state=0).fit(data, data_target)
print "Parametry: 0, 1, 2, 3"
print np.mean(cross_val_score(kmeans, d, data_target, cv=5))



#!/usr/bin/ev python

import numpy as np
import pylab
from sklearn.decomposition import PCA



#Pobieranie danych z pliku
all_data = pylab.loadtxt("wine.data", delimiter=",")
names = ["Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
data = all_data[:,1:]
classes = all_data[:,:1]
classes_name = ['1','2','3']


x = range(1, 14)
y = []
for i in x:
	pca = PCA(n_components=i)
	pca.fit(data, classes)
	y.append(np.mean(pca.explained_variance_ratio_))
	print np.mean(pca.explained_variance_ratio_)

pca = PCA(n_components=2)
pca.fit(data[:,:2],classes)
print np.mean(pca.explained_variance_ratio_)
pca.fit(data[:,1:3],classes)
print np.mean(pca.explained_variance_ratio_)

pylab.plot(x, y)
pylab.title('Zaleznosc skumulowanego odchylenia standardowego od liczby zmiennych')
pylab.xlabel('Liczba zmiennych')
pylab.ylabel('Skumulowane odchylenie standardowe')
pylab.grid(True)
pylab.show()

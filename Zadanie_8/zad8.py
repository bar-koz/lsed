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
	pca = PCA()
	pca.fit(data[:,:i])
	y.append(np.sum(np.sqrt(pca.explained_variance_)))
	print np.sum(np.sqrt(pca.explained_variance_))

pca = PCA()
pca.fit(data[:,:2])
print np.sum(np.sqrt(pca.explained_variance_))
pca.fit(data[:,1:3])
print np.sum(np.sqrt(pca.explained_variance_))

#Rysowanie wykresu
pylab.plot(x, y)
pylab.title('Zaleznosc skumulowanego odchylenia standardowego od liczby zmiennych')
pylab.xlabel('Liczba zmiennych')
pylab.ylabel('Skumulowane odchylenie standardowe')
pylab.grid(True)
pylab.show()

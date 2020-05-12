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
previous = 0
for i in x:
	pca = PCA(n_components=i)
	pca.fit(data)
	tmp = pca.explained_variance_
	previous += np.sqrt(tmp[i-1])
	y.append(previous)
	print np.sqrt(tmp[i-1])

pca = PCA()
pca.fit(data[:,:2])
print np.sqrt(pca.explained_variance_)
pca.fit(data[:,1:3])
print np.sqrt(pca.explained_variance_)

#Rysowanie wykresu
pylab.plot(x, y)
pylab.title('Zaleznosc skumulowanego odchylenia standardowego od liczby zmiennych')
pylab.xlabel('Liczba zmiennych')
pylab.ylabel('Skumulowane odchylenie standardowe')
pylab.grid(True)
pylab.show()

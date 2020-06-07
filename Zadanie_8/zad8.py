#!/usr/bin/ev python

import numpy as np
import pylab
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



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
	#print np.sqrt(tmp[i-1])
#Obliczenia nowych zmiennych
pca = PCA(n_components=13)
pca.fit(data)
principalComponents = pca.fit_transform(data)





#Rysowanie wykresu
pylab.plot(x, y)
pylab.title('Zaleznosc skumulowanego odchylenia standardowego od liczby zmiennych')
pylab.xlabel('Liczba zmiennych')
pylab.ylabel('Skumulowane odchylenie standardowe')
pylab.grid(True)
pylab.show()

#Rysowanie wykresu 1 i 2 skladowa
pylab.scatter(principalComponents[:,0], principalComponents[:,1])
pylab.title('Punkty nowych zmiennych')
pylab.xlabel('Pierwsza skladowa')
pylab.ylabel('Druga skladowa')
pylab.grid(True)
pylab.show()

#Rysowanie wykresu 2 i 3 skladowa
pylab.scatter(principalComponents[:,1], principalComponents[:,2])
pylab.title('Punkty nowych zmiennych')
pylab.xlabel('Druga skladowa')
pylab.ylabel('Trzecia skladowa')
pylab.grid(True)
pylab.show()


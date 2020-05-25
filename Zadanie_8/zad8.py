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
#Obliczenia nowych zmiennych
z1 = pca.components_[0]
z2 = pca.components_[1]
z3 = pca.components_[2]
print z1
x1 = []
x2 = []
x3 = []
for i in range(0, 178):
	x1.append(np.sum(z1*data[i:i+1,:]))
	x2.append(np.sum(z2*data[i:i+1,:]))
	x3.append(np.sum(z3*data[i:i+1,:]))


#Rysowanie wykresu
pylab.plot(x, y)
pylab.title('Zaleznosc skumulowanego odchylenia standardowego od liczby zmiennych')
pylab.xlabel('Liczba zmiennych')
pylab.ylabel('Skumulowane odchylenie standardowe')
pylab.grid(True)
pylab.show()

#Rysowanie wykresu 1 i 2 skladowa
pylab.scatter(x1, x2)
pylab.title('Punkty nowych zmiennych')
pylab.xlabel('Pierwsza skladowa')
pylab.ylabel('Druga skladowa')
pylab.grid(True)
pylab.show()

#Rysowanie wykresu 2 i 3 skladowa
pylab.scatter(x2, x3)
pylab.title('Punkty nowych zmiennych')
pylab.xlabel('Druga skladowa')
pylab.ylabel('Trzecia skladowa')
pylab.grid(True)
pylab.show()


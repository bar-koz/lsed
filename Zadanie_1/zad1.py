#!/usr/bin/ev python

import numpy as np
import pylab
from sklearn.naive_bayes import GaussianNB

# Dane
S = np.array([[4.,0.],[0.,4.]])
m1 = [-3.,-1.]
m2 = [2.,2.]
# Liczba obserwacji w klasach
n1 = 40
n2 = 30
# Generowanie obserwacji
a1 = np.random.multivariate_normal(m1,S,n1)
a2 = np.random.multivariate_normal(m2,S,n2)
# Tablica wszystkich obserwacji
features = np.concatenate((a1, a2))
# Tworzenie tablicy odzwierciedlajacej przynaleznosc obserwacji do klas
label_a1 = np.zeros(n1)
label_a2 = np.ones(n2)
label = np.concatenate((label_a1,label_a2))
# Tworzenie klasyfikatora
model = GaussianNB()
# Dopasowywanie klasyfikatora do danych
model.fit(features, label)
# Przykladowe przewidywanie przynaleznosci do klas
predicted= model.predict([a2[1]])
print "Predicted Value:", predicted


# Obszary klas wyznaczone przez klasyfikator Bayesowski
x_a1, y_a1 = [],[]
x_a2, y_a2 = [],[]
for x in np.arange(-10.,10.,0.1):
	for y in np.arange(-10.,10.,0.1):
		predicted= model.predict([x,y])
		if predicted == 0:
			x_a1.append(x)
			y_a1.append(y)
		if predicted == 1:
			x_a2.append(x)
			y_a2.append(y)

# Rysowanie obszaru klas wyznaczonych przez klasyfikator Bayesowski
pylab.plot(x_a1, y_a1, '.', label='Klasa 1',  markersize=10, alpha=0.5)
pylab.plot(x_a2, y_a2, '.', label='Klasa 2',  markersize=10, alpha=0.5)
# Rysowanie obserwacji
pylab.plot(a1[:,0], a1[:,1], '.', label='a1')
pylab.plot(a2[:,0], a2[:,1], '.', label='a2')
pylab.title('Naiwny klasyfikator Bayesowski')
pylab.legend()
pylab.grid(True)
pylab.show()

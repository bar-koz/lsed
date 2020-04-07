#!/usr/bin/ev python

import numpy as np
import pylab

# Macierz kowariancji
S = np.eye(2)
# Macierze srednich dla wszystkich klas
m1 = np.array( [ [-1.],[1.] ] )
m2 = np.array( [ [2.],[4.] ] )
m3 = np.array( [ [-2.],[2.] ] )

# Liczebnosc klas
n1 = 30.
n2 = 30.
n3 = 30.
# Ogolna liczebnosc
n = n1 + n2 + n3
# Liczba klas
g = 3.
# Srednia dla wszystkich punktow
m = (m1 + m2 + m3) / 3.
# Macierz zmiennosci miedzygropowej
B = (np.dot(n1*(m1 - m), (m1 - m).T) + np.dot(n2*(m2 - m), (m2 - m).T) + np.dot(n3*(m3 - m), (m3 - m).T)) / (g - 1)
# Macierz zmiennosci wewnartzgrupowej
W = 1. / (n - g) * ((n1 - 1.) * S + (n2 - 1.) * S + (n3 - 1.) * S)
# Macierz pomocnicza
U = np.dot(np.linalg.inv(W), B)
# Macierze wartosci wlasnych i wektorow wlasnych
value, vector = np.linalg.eig(U)
# Wektor wlasny odpowiadajacy najwiekszej wartosci wlasnej
A = vector[np.argmax(value),:]

# Tworzenie tablicy punktow kazdej klasy z uwzglednieniem ich srednich
a1 = np.random.rand(30,2)
a2 = np.random.rand(30,2)
a3 = np.random.rand(30,2)
a1 = a1 * m1.T + m1.T / 2
a2 = a2 * m2.T + m2.T / 2
a3 = a3 * m3.T + m3.T / 2

# Tworzenie wykresu prostej kierunku a
x = range(-4,5)
y = []
for i in x:
	y.append(A[1] / A[0] * i)
#Rzutowanie na prosta
a1_z = []
a2_z = []
a3_z = []
a = A[1] / A[0]
def B(ax):
	return ax[1] + 1/a * ax[0]
for i in a1:
	b = B(i)
	a1_z.append([b/(a+1./a),a*b/(a+1./a)])
a1_z = np.asarray(a1_z)
for i in a2:
	b = B(i)
	a2_z.append([b/(a+1./a),a*b/(a+1./a)])
a2_z = np.asarray(a2_z)
for i in a3:
	b = B(i)
	a3_z.append([b/(a+1./a),a*b/(a+1./a)])
a3_z = np.asarray(a3_z)

a1 = np.concatenate((a1,a1_z))
a2 = np.concatenate((a2,a2_z))
a3 = np.concatenate((a3,a3_z))

# Rysowanie kierunku A
pylab.plot(x, y)
# Rysowanie punktow z klas
pylab.plot(a1[:,0], a1[:,1], '.', label='Klasa 1')
pylab.plot(a2[:,0], a2[:,1], '.', label='Klasa 2')
pylab.plot(a3[:,0], a3[:,1], '.', label='Klasa 3')
pylab.title('Kierunek rzutu a')
pylab.legend()
pylab.grid(True)
pylab.show()

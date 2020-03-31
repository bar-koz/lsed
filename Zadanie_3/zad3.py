#!/usr/bin/ev python

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pylab

def ACC(CM):
	TN = CM[0,0]
	FP = CM[0,1]
	FN = CM[1,0]
	TP = CM[1,1]
	return float((TP + TN)) / float((TP + TN + FP + FN))


# Dane
S1 = np.array([[4,2],[2,4]])
S2 = np.array([[4,2],[2,2]])
m1 = np.array([-1,-1])
m2 = np.array([2,2])
# Liczba obserwacji w klasach
n1 = 30
n2 = 20
# Generowanie obserwacji
a1 = np.random.multivariate_normal(m1,S1,n1)
a2 = np.random.multivariate_normal(m2,S2,n2)
#Skladanie danych
data = np.concatenate((a1, a2))
n1_vector = np.zeros(n1)
n2_vector = np.ones(n2)
classes = np.concatenate((n1_vector,n2_vector))
#Dane testowe
t1 = np.random.multivariate_normal(m1,S1,10)
t2 = np.random.multivariate_normal(m2,S2,5)
test_data = np.concatenate((t1, t2))
test_classes = np.concatenate((np.zeros(10), np.ones(5)))

#Obliczenia ACC, TN i TP dla k_nn 1 - 21
x = range(1,22)
y = []
TP, TN = [], []
test_y, test_TP, test_TN = [], [], []
for i in x:
	#Metoda k_nn
	neigh = KNeighborsClassifier(n_neighbors=i)
	neigh.fit(data, classes)
	pred_data = neigh.predict(data)
	#Macierz pomylek
	conf_matrix = confusion_matrix(classes, pred_data, labels=[0,1])
	y.append(ACC(conf_matrix))
	TP.append(conf_matrix[1,1])
	TN.append(conf_matrix[0,0])
	pred_data_test = neigh.predict(test_data)
	conf_matrix_test = confusion_matrix(test_classes, pred_data_test, labels=[0,1])
	test_y.append(ACC(conf_matrix_test))
	test_TP.append(conf_matrix_test[1,1])
	test_TN.append(conf_matrix_test[0,0])

pylab.plot(x, y, label='Dane wejsciowe = Testowe')
pylab.plot(x, test_y, label='Dane wejsciowe =/= Testowe')
pylab.title('Skutecznosc Klasyfikatora')
pylab.xlabel('k_nn')
pylab.ylabel('ACC')
pylab.legend()
pylab.grid(True)
pylab.show()

pylab.plot(x, TP, label='Dane wejsciowe = Testowe')
pylab.title('TRUE POSITIVE')
pylab.xlabel('k_nn')
pylab.ylabel('TP')
pylab.legend()
pylab.grid(True)
pylab.show()

pylab.plot(x, TN, label='Dane wejsciowe = Testowe')
pylab.title('TRUE NEGATIVE')
pylab.xlabel('k_nn')
pylab.ylabel('TN')
pylab.legend()
pylab.grid(True)
pylab.show()

pylab.plot(x, test_TP, label='Dane wejsciowe =/= Testowe')
pylab.title('TRUE POSITIVE')
pylab.xlabel('k_nn')
pylab.ylabel('TP')
pylab.legend()
pylab.grid(True)
pylab.show()

pylab.plot(x, test_TN, label='Dane wejsciowe =/= Testowe')
pylab.title('TRUE NEGATIVE')
pylab.xlabel('k_nn')
pylab.ylabel('TN')
pylab.legend()
pylab.grid(True)
pylab.show()






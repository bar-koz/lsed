#!/usr/bin/ev python

import numpy as np
import pylab
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

from sklearn.externals.six import StringIO  
import pydot
import math 
import pylab




#Pobieranie danych z pliku
all_data = pylab.loadtxt("wine.data", delimiter=",")
names = ["Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
data = all_data[:,1:]
classes = all_data[:,:1]
classes_name = ['1','2','3']
#Liczba obserwacji
n1 = 59
n2 = 71
n3 = 48
n = n1 + n2 + n3


treeClass = tree.DecisionTreeClassifier()#max_depth=2
treeClass = treeClass.fit(data, classes)

#Rysowanie pelnego drzewa 
def RYS(treeClass, name, classes_name):
	dot_data = StringIO() 
	tree.export_graphviz(treeClass, out_file=dot_data,
		feature_names=names,
		class_names=classes_name,
		filled=True, rounded=True,
		special_characters=True) 
	graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
	graph.write_pdf(name) 

RYS(treeClass,"tree.png",classes_name)


print ("ACC w przypadku powtornego podstawienia")
print (treeClass.score(data, classes))

#Transformacja danych dla funkcji cross_val_score
classes = all_data[:,0]
print ("ACC w przypdaku kroswalidacji")
print (np.mean(cross_val_score(treeClass, data, classes, cv=5)))

#Wybor optymalnego drzewa
#Nie ma w wersji 0.17.1 tej funkcji
test = treeClass.cost_complexity_pruning_path(data, classes)
op_tree = treeClass.decision_path(data)

def beta_i(array, i):
	if i == 0:
		return array[i]
	else:
		return math.sqrt(array[i-1]*array[i])

score = []
for i in test.ccp_alphas:
	treeClassb = tree.DecisionTreeClassifier(ccp_alpha=i)
	treeClassb = treeClassb.fit(data, classes)
	score.append(np.mean(cross_val_score(treeClassb, data, classes, cv=5)))

best_cp = beta_i(test.ccp_alphas, np.argmax(score))

treeClass_best = tree.DecisionTreeClassifier(ccp_alpha=best_cp)
treeClass_best = treeClass_best.fit(data, classes)
print ("ACC w przypadku powtornego podstawienia - optymalne drzewo")
print (treeClass_best.score(data, classes))

print ("ACC w przypdaku kroswalidacji - optymalne drzewo")
print (np.mean(cross_val_score(treeClass, data, classes, cv=5)))
#Rysowanie optymalnego drzewa
RYS(treeClass_best,"tree_best.png",classes_name)

#Obliczenia do wykresu skutecznosci i rozmiaru drzewa w zaleznosci od liczby parametrow
acc_table_full = []
acc_table_optimal = []
n_leaves_full = []
n_leaves_optimal = []
for i in range(1,14):
	treeClass_tmp = tree.DecisionTreeClassifier()
	treeClass_tmp = treeClass_tmp.fit(data[:,:i], classes)
	acc_table_full.append([treeClass_tmp.score(data[:,:i], classes),np.mean(cross_val_score(treeClass_tmp, data[:,:i], classes))])
	n_leaves_full.append(treeClass_tmp.get_n_leaves())
	#Optmalne drzewo
	test = treeClass_tmp.cost_complexity_pruning_path(data[:,:i], classes)
	score = []
	for k in test.ccp_alphas:
		treeClassb = tree.DecisionTreeClassifier(ccp_alpha=k)
		treeClassb = treeClassb.fit(data[:,:i], classes)
		score.append(np.mean(cross_val_score(treeClassb, data[:,:i], classes, cv=5)))
	best_cp = beta_i(test.ccp_alphas, np.argmax(score))
	treeClass_best = tree.DecisionTreeClassifier(ccp_alpha=best_cp)
	treeClass_best = treeClass_best.fit(data[:,:i], classes)
	acc_table_optimal.append([treeClass_best.score(data[:,:i], classes),np.mean(cross_val_score(treeClass_best, data[:,:i], classes))])
	n_leaves_optimal.append(treeClass_best.get_n_leaves())

#Rysowanie wykresow
x = range(1,14)
acc_table_full = np.asarray(acc_table_full)
acc_table_optimal = np.asarray(acc_table_optimal)

pylab.plot(x, acc_table_full[:,0], label='Dane wejsciowe = Testowe - Pelne drzewo')
pylab.plot(x, acc_table_full[:,1], label='Kroswalidacja - Pelne drzewo')
pylab.plot(x, acc_table_optimal[:,0], label='Dane wejsciowe = Testowe - Optymalne drzewo')
pylab.plot(x, acc_table_optimal[:,1], label='Kroswalidacja - Optymalne drzewo')
pylab.title('Skutecznosc Drzewa')
pylab.xlabel('Liczba zmiennych')
pylab.ylabel('ACC')
pylab.legend()
pylab.grid(True)
pylab.show()

n_leaves_full = np.asarray(n_leaves_full)
n_leaves_optimal = np.asarray(n_leaves_optimal)

pylab.plot(x, n_leaves_full, label='Pelne drzewo')
pylab.plot(x, n_leaves_optimal, label='Optymalne drzewo')
pylab.title('Roznice rozmiaru drzewa pelnego i optymalnego')
pylab.xlabel('Liczba zmiennych')
pylab.ylabel('Rozmiar(liczba lisci)')
pylab.legend()
pylab.grid(True)
pylab.show()



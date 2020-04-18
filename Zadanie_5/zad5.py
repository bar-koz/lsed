#!/usr/bin/ev python

import numpy as np
import pylab
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Tworzenie zbioru danych
data = load_iris()
data_target = data.target
data = data.data
# Dzielenie danych na PU i PT
X_train, X_test, Target_train, Target_test = train_test_split(data, data_target, test_size = 0.2, random_state = 42)

n_classificators = [1, 2, 5, 10, 20, 50]
mistakes = []
for i in n_classificators:
	bagging = BaggingClassifier(LinearDiscriminantAnalysis(), n_estimators = i)
	bagging_i = bagging.fit(X_train, Target_train)
	bagging_full = bagging.fit(data, data_target)
	pred_i = bagging_i.predict(X_test)
	pred_full = bagging_full.predict(data)
	mistakes.append((Target_test.size - np.sum(pred_i == Target_test), data_target.size - np.sum(pred_full == data_target)))

mistakes = np.asarray(mistakes)
# Rysowanie wykresu
pylab.plot(n_classificators, mistakes[:,1], label='Pierwotny zbior')
pylab.plot(n_classificators, mistakes[:,0], label='Zmodyfikowany zbior')
pylab.title('Wykres blednych klasyfikacji')
pylab.xlabel('Liczba klasyfikatorow')
pylab.ylabel('Liczba blednych klasyfikacji')
pylab.legend()
pylab.grid(True)
pylab.show()


#!/usr/bin/ev python

import numpy as np
import pylab
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score


# Macierz kowariancji
S = np.eye(2)
# Macierze srednich dla wszystkich klas
m1 = [-1.,1.]
m2 = [2.,4.]
m3 = [-2.,2.]

# Liczebnosc klas
n1 = 30
n2 = 30
n3 = 30
# Ogolna liczebnosc
n = n1 + n2 + n3

# Generowanie obserwacji
a1 = np.random.multivariate_normal(m1,S,n1)
a2 = np.random.multivariate_normal(m2,S,n2)
a3 = np.random.multivariate_normal(m3,S,n2)
y1 = np.zeros(n1)
y2 = np.ones(n2)
y3 = np.ones(n3)*2
# Laczenie danych
a = np.concatenate((a1, a2))
a = np.concatenate((a, a3))
y = np.concatenate((y1, y2))
y = np.concatenate((y, y3))

C = np.arange(0.1, 1.1, 0.1)
score_svm = []
score_lda = []

lda = LinearDiscriminantAnalysis()
lda.fit(a, y)

for i in C:
	svm = LinearSVC(C=i)
	svm.fit(a, y)
	score_svm.append([svm.score(a, y), np.mean(cross_val_score(svm, a, y, cv=5))])
	score_lda.append([lda.score(a, y), np.mean(cross_val_score(lda, a, y, cv=5))])

score_svm = np.asarray(score_svm)
score_lda = np.asarray(score_lda)


pylab.plot(C, score_svm[:,0], label='SVM - powtorne podstawienie')
pylab.plot(C, score_svm[:,1], label='SVM - kroswalifacja')
pylab.plot(C, score_lda[:,0], label='LDA - powtorne podstawienie')
pylab.plot(C, score_lda[:,1], label='LDA - kroswalifacja')
pylab.title('Skutecznosc klasyfikatorow')
pylab.xlabel('C')
pylab.ylabel('ACC')
pylab.legend()
pylab.grid(True)
pylab.show()



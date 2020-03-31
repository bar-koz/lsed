#!/usr/bin/ev python

import numpy as np
import pylab
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_predict

def ACC(CM):
	TN = CM[0,0]
	FP = CM[0,1]
	FN = CM[1,0]
	TP = CM[1,1]
	return float((TP + TN)) / float((TP + TN + FP + FN))
def FPR(CM):
	TN = float(CM[0,0])
	FP = float(CM[0,1])
	return FP / (FP + TN)
def TPR(CM):
	FN = float(CM[1,0])
	TP = float(CM[1,1])
	return TP / (TP + FN)
def CM(PreCM):
	TP = np.trace(PreCM)
	TN = np.sum(PreCM)
	FN = TN - TP
	FP = FN
	CM = [[TN,FP],[FN,TP]]
	return np.asarray(CM)


#Pobieranie danych z pliku
all_data = pylab.loadtxt("wine.data", delimiter=",")
names = ["Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
data = all_data[:,1:]
classes = all_data[:,:1]
#Liczba obserwacji
n1 = 59
n2 = 71
n3 = 48
n = n1 + n2 + n3

#Metoda powtornego podstawienia
##############################
def ALL(data):
	#Naive Bayes
	# Tworzenie klasyfikatora
	model = GaussianNB()
	# Dopasowywanie klasyfikatora do danych
	model.fit(data, classes.ravel())

	#LDA
	# Tworzenie klasyfikatora
	lda = LinearDiscriminantAnalysis()
	# Dopasowywanie klasyfikatora do danych
	lda.fit(data,classes.ravel())

	#QDA
	# Tworzenie klasyfikatora
	qda = QuadraticDiscriminantAnalysis()
	# Dopasowywanie klasyfikatora do danych
	qda.fit(data, classes.ravel())

	
	pred_NB = model.predict(data)
	PreCM_NB = confusion_matrix(classes, pred_NB, labels=[1,2,3])

	pred_LDA = lda.predict(data)
	PreCM_LDA = confusion_matrix(classes, pred_LDA, labels=[1,2,3])

	pred_QDA = qda.predict(data)
	PreCM_QDA = confusion_matrix(classes, pred_QDA, labels=[1,2,3])
	#Macierz Pomylek
	CM_NB = CM(PreCM_NB)
	CM_LDA = CM(PreCM_LDA)
	CM_QDA = CM(PreCM_QDA)

	print "Powtorne podstawienie"
	print("        ACC      TP    TN    TPR      FPR")
	print("NB      %6.4f   %d   %d   %6.4f   %6.4f" % (ACC(CM_NB), CM_NB[1,1], CM_NB[0,0], TPR(CM_NB), FPR(CM_NB)))
	print("LDA     %6.4f   %d   %d   %6.4f   %6.4f" % (ACC(CM_LDA), CM_LDA[1,1], CM_LDA[0,0], TPR(CM_LDA), FPR(CM_LDA)))
	print("QDA     %6.4f   %d   %d   %6.4f   %6.4f" % (ACC(CM_QDA), CM_QDA[1,1], CM_QDA[0,0], TPR(CM_QDA), FPR(CM_QDA)))
###############################
print "Wszystkie parametry"
ALL(data)

print "2 parametry"
data_2 = data[:,:2]
ALL(data_2)
print "5 parametrow"
data_5 = data[:,:5]
ALL(data_5)
print "10 parametrow"
data_10 = data[:,:10]
ALL(data_10)



#Rozdzielone dane na PU,PW i PT
##############################
def ALL_2(data):

	PW, PU, PW_classes, PU_classes = train_test_split(data, classes, test_size=0.5, random_state=1)

	PT, PW, PT_classes, PW_classes = train_test_split(PW, PW_classes, test_size=0.5, random_state=2)

	

	#Naive Bayes
	# Tworzenie klasyfikatora
	model = GaussianNB()
	# Dopasowywanie klasyfikatora do danych
	model.fit(PU, PU_classes.ravel())

	#LDA
	# Tworzenie klasyfikatora
	lda = LinearDiscriminantAnalysis()
	# Dopasowywanie klasyfikatora do danych
	lda.fit(PU, PU_classes.ravel())

	#QDA
	# Tworzenie klasyfikatora
	qda = QuadraticDiscriminantAnalysis()
	# Dopasowywanie klasyfikatora do danych
	qda.fit(PU, PU_classes.ravel())

	pred_NB = model.predict(PW)
	PreCM_NB = confusion_matrix(PW_classes, pred_NB, labels=[1,2,3])

	pred_LDA = lda.predict(PW)
	PreCM_LDA = confusion_matrix(PW_classes, pred_LDA, labels=[1,2,3])

	pred_QDA = qda.predict(PW)
	PreCM_QDA = confusion_matrix(PW_classes, pred_QDA, labels=[1,2,3])
	#Macierz Pomylek
	CM_NB = CM(PreCM_NB)
	CM_LDA = CM(PreCM_LDA)
	CM_QDA = CM(PreCM_QDA)

	print "Podzielony zbior danych"
	print("        ACC      TP    TN    TPR      FPR")
	print("NB      %6.4f   %d    %d   %6.4f   %6.4f" % (ACC(CM_NB), CM_NB[1,1], CM_NB[0,0], TPR(CM_NB), FPR(CM_NB)))
	print("LDA     %6.4f   %d    %d   %6.4f   %6.4f" % (ACC(CM_LDA), CM_LDA[1,1], CM_LDA[0,0], TPR(CM_LDA), FPR(CM_LDA)))
	print("QDA     %6.4f   %d    %d   %6.4f   %6.4f" % (ACC(CM_QDA), CM_QDA[1,1], CM_QDA[0,0], TPR(CM_QDA), FPR(CM_QDA)))
###############################

ALL_2(data_2)

#Kroswalidacja
##############################
def ALL_3(data):
	#Transformacja danych dla funkcji cross_val_score
	classes = all_data[:,0]

	#LDA
	# Tworzenie klasyfikatora
	lda = LinearDiscriminantAnalysis()
	#Cross-validacja
	pred_LDA = cross_val_predict(lda, data, classes, cv=5)
	PreCM_LDA = confusion_matrix(classes, pred_LDA, labels=[1,2,3])
	#Macierz Pomylek
	CM_LDA = CM(PreCM_LDA)

	
	print "Wynik z kroswalidacja"
	print("        ACC      TP    TN    TPR      FPR")
	print("LDA     %6.4f   %d   %d  %6.4f   %6.4f" % (ACC(CM_LDA), CM_LDA[1,1], CM_LDA[0,0], TPR(CM_LDA), FPR(CM_LDA)))
###############################

ALL_3(data_2)



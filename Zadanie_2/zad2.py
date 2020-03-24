#!/usr/bin/ev python

import numpy as np
import pylab
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

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

	pred_NB = []
	for i in range(n):
		predicted = model.predict([data[i]])
		if predicted == 1:
			pred_NB.append(1)
		if predicted == 2:
			pred_NB.append(2)
		if predicted == 3:
			pred_NB.append(3)
	PreCM_NB = confusion_matrix(classes, pred_NB, labels=[1,2,3])

	pred_LDA = []
	for i in range(n):
		predicted = lda.predict([data[i]])
		if predicted == 1:
			pred_LDA.append(1)
		if predicted == 2:
			pred_LDA.append(2)
		if predicted == 3:
			pred_LDA.append(3)
	PreCM_LDA = confusion_matrix(classes, pred_LDA, labels=[1,2,3])

	pred_QDA = []
	for i in range(n):
		predicted = qda.predict([data[i]])
		if predicted == 1:
			pred_QDA.append(1)
		if predicted == 2:
			pred_QDA.append(2)
		if predicted == 3:
			pred_QDA.append(3)
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

	PU1 = data[:n1/2,:2]
	PU2 = data[n1:n1+n2/2,:2]
	PU3 = data[n1+n2:n1+n2+n3/2,:2]
	PU = np.concatenate((PU1,PU2))
	PU = np.concatenate((PU,PU3))

	PW1 = data[n1/2:3*n1/4,:2]
	PW2 = data[n1+n2/2:n1+3*n2/4,:2]
	PW3 = data[n1+n2+n3/2:n1+n2+3*n3/4,:2]
	PW = np.concatenate((PW1,PW2))
	PW = np.concatenate((PW,PW3))

	PT1 = data[3*n1/4:n1,:2]
	PT2 = data[n1+3*n2/4:n1+n2,:2]
	PT3 = data[n1+n2+3*n3/4:n,:2]
	PT = np.concatenate((PT1,PT2))
	PT = np.concatenate((PT,PT3))

	PU_classes = np.ones(n1/2)
	PU_classes = np.concatenate((PU_classes, np.ones(n2/2) * 2))
	PU_classes = np.concatenate((PU_classes, np.ones(n3/2) * 3))

	PW_classes = np.ones(n1/4)
	PW_classes = np.concatenate((PW_classes, np.ones(n2/4) * 2))
	PW_classes = np.concatenate((PW_classes, np.ones(n3/4) * 3))
	
	PT_classes = PW_classes

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

	pred_NB = []
	for i in range(n/4-1):
		predicted = model.predict([PW[i]])
		if predicted == 1:
			pred_NB.append(1)
		if predicted == 2:
			pred_NB.append(2)
		if predicted == 3:
			pred_NB.append(3)
	PreCM_NB = confusion_matrix(PW_classes, pred_NB, labels=[1,2,3])

	pred_LDA = []
	for i in range(n/4-1):
		predicted = lda.predict([PW[i]])
		if predicted == 1:
			pred_LDA.append(1)
		if predicted == 2:
			pred_LDA.append(2)
		if predicted == 3:
			pred_LDA.append(3)
	PreCM_LDA = confusion_matrix(PW_classes, pred_LDA, labels=[1,2,3])

	pred_QDA = []
	for i in range(n/4-1):
		predicted = qda.predict([PW[i]])
		if predicted == 1:
			pred_QDA.append(1)
		if predicted == 2:
			pred_QDA.append(2)
		if predicted == 3:
			pred_QDA.append(3)

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

	PU1 = data[:n1/2,:2]
	PU2 = data[n1:n1+n2/2,:2]
	PU3 = data[n1+n2:n1+n2+n3/2,:2]
	PU = np.concatenate((PU1,PU2))
	PU = np.concatenate((PU,PU3))

	PW1 = data[n1/2:3*n1/4,:2]
	PW2 = data[n1+n2/2:n1+3*n2/4,:2]
	PW3 = data[n1+n2+n3/2:n1+n2+3*n3/4,:2]
	PW = np.concatenate((PW1,PW2))
	PW = np.concatenate((PW,PW3))

	PT1 = data[3*n1/4:n1,:2]
	PT2 = data[n1+3*n2/4:n1+n2,:2]
	PT3 = data[n1+n2+3*n3/4:n,:2]
	PT = np.concatenate((PT1,PT2))
	PT = np.concatenate((PT,PT3))

	PU_classes = np.ones(3*n1/4)
	PU_classes = np.concatenate((PU_classes, np.ones(3*n2/4) * 2))
	PU_classes = np.concatenate((PU_classes, np.ones(3*n3/4) * 3))

	PT_classes = np.ones(n1/4)
	PT_classes = np.concatenate((PT_classes, np.ones(n2/4) * 2))
	PT_classes = np.concatenate((PT_classes, np.ones(n3/4) * 3))
		
	PU = np.concatenate((PU, PW))


	#LDA
	# Tworzenie klasyfikatora
	lda = LinearDiscriminantAnalysis()
	# Dopasowywanie klasyfikatora do danych
	lda.fit(PU, PU_classes.ravel())

	pred_LDA = []
	for i in range(n/4-1):
		predicted = lda.predict([PT[i]])
		if predicted == 1:
			pred_LDA.append(1)
		if predicted == 2:
			pred_LDA.append(2)
		if predicted == 3:
			pred_LDA.append(3)
	PreCM_LDA = confusion_matrix(PT_classes, pred_LDA, labels=[1,2,3])
	#Macierz Pomylek
	CM_LDA = CM(PreCM_LDA)

	
	print "Wynik z kroswalidacja"
	print("        ACC      TP    TN    TPR      FPR")
	print("LDA     %6.4f   %d    %d   %6.4f   %6.4f" % (ACC(CM_LDA), CM_LDA[1,1], CM_LDA[0,0], TPR(CM_LDA), FPR(CM_LDA)))
###############################

ALL_3(data_2)



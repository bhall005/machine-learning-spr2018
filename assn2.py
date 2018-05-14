# Brennan Hall - 861198641
# CS 171 18S
# Assignment 2

import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
import copy
from pandas.plotting import scatter_matrix
from random import shuffle

# Global Lists/Variables

allPts = []

# ----- QUESTION 0 -----
# Reads and parses both iris and wine data into meaningful lists
def readNParse(bcFile):
	global allPts

	# Split the data into line chunks
	with open(bcFile, 'r') as bData:
		bChunks = bData.read().splitlines()

	# Parse iris data chunks into data points
	for m in range(len(bChunks)):
		temp = []
		badData = False
		splitChunk = bChunks[m].split(',')
		for q in range(11):
			if q == 0:
				continue
			else:
				# Remove data with missing attributes
				if splitChunk[q] == '?':
					badData = True
					break
				else:
					temp.append(float(splitChunk[q]))
		if badData == False:
			allPts.append(temp)


# ----- QUESTION 1 -----

# Computes the Lp norm from the given sets of data points using the given
# value of p
def distance(x, y, p):
	if len(x) != len(y):
		print 'distance error: x and y are not of equal length'
		exit(1)

	tmp = 0
	for i in range(len(x)):
		tmp += pow(abs(x[i]-y[i]), p)
	return pow(tmp, (1.0/float(p)))

# Takes in training data and test data and computes 'k' nearest neighbor
# using both sets of data to classify binary attributes
# Returns a 1D list of the test data with predicted labels
def knn_classifier(x_test, x_train, y_train, k, p):
	y_pred = copy.deepcopy(x_test)
	for i in range(len(x_test)):
		bCount = 0
		mCount = 0
		rawDistances = []
		# Calculate distances between point i and all training points
		for j in range(len(x_train)):
			if x_test[i] == x_train[j]:
				continue;
			rawDistances.append((distance(x_test[i][:-1], x_train[j][:-1], p), j))
		# Select the k-nearest neighbors
		finalDist = sorted(rawDistances)[:k]
		# Tally up the classes of the nearest neighbors
		for n in finalDist:
			if x_train[n[1]][9] == 2.0:
				bCount += 1
			else:
				mCount += 1
		if bCount > mCount:
			y_pred[i].append('Benign')
		else:
			y_pred[i].append('Malignant')
		if y_pred[i][9] == 2.0:
			y_pred[i][9] = 'Benign'
		else:
			y_pred[i][9] = 'Malignant'

	return y_pred


# Calculates the percent accuracy of the k-NN classifier
def accuracyCalc(y_pred, printBool):
	correctCnt = 0
	total = len(y_pred)
	for i in range(total):
		if printBool:
			print('Predicted: ' + y_pred[i][10] + '  Actual: ' + y_pred[i][9])
		if y_pred[i][9] == y_pred[i][10]:
			correctCnt += 1

	accuracy = (float(correctCnt)/float(total)) * 100
	print ('Accuracy: ' + str(accuracy) + '%')
	return accuracy


# ----- QUESTION 2 -----

# Calculates the percent specificity of the k-NN classifier
def specificityCalc(y_pred):
	TP_cnt = 0
	P_cnt = 0
	for i in range(len(y_pred)):
		if y_pred[i][9] == 'Malignant':
			P_cnt += 1
			if y_pred[i][10] == 'Malignant':
				TP_cnt += 1
	specificity = (float(TP_cnt)/float(P_cnt)) * 100
	print ('Specificity: ' + str(specificity) + '%')
	return specificity

# Calculates the percent sensitivity of the k-NN classifier
def sensitivityCalc(y_pred):
	TN_cnt = 0
	N_cnt = 0
	for i in range(len(y_pred)):
		if y_pred[i][9] == 'Benign':
			N_cnt += 1
			if y_pred[i][10] == 'Benign':
				TN_cnt += 1
	sensitivity = (float(TN_cnt)/float(N_cnt)) * 100
	print ('Sensitivity: ' + str(sensitivity) + '%')
	return sensitivity

# Performs 10-fold cross validation of the k-NN classifier
# Uses different values for k and p
# Plots graphs to track performance of different k and p values
def crossValidation(dataSet, classifierName):
	dataFolds = []
	boundConst = int(math.floor(len(dataSet) * 0.1))
	lBound = 0
	uBound = boundConst

	# Randomly shuffle the data
	shuffle(dataSet)

	# Partition data into 10 folds
	while (len(dataFolds) != 10):
		dataFolds.append(dataSet[lBound:uBound])
		lBound += boundConst
		uBound += boundConst

	# Disgusting amount of lists necessary to create separate graphs
	accMeanP1 = []
	accMeanP2 = []
	specMeanP1 = []
	specMeanP2 = []
	sensMeanP1 = []
	sensMeanP2 = []
	accStdP1 = []
	accStdP2 = []
	specStdP1 = []
	specStdP2 = []
	sensStdP1 = []
	sensStdP2 = []
	kList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

	# MAIN LOOP - Performs hundreds of k-NN iterations
	for p in {1, 2}:
		for k in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}:
			accuracyList = []
			specificityList = []
			sensitivityList = []
			for i in dataFolds:
				x_test = i
				x_train = []
				for j in dataFolds:
					if i != j:
						x_train += j
				y_pred = []

				if classifierName == 'kNN':
					y_pred = knn_classifier(x_test, x_train, x_test, k, p)

				accuracyList.append(accuracyCalc(y_pred, False))
				specificityList.append(specificityCalc(y_pred))
				sensitivityList.append(sensitivityCalc(y_pred))
			if p == 1:
				accMeanP1.append(np.mean(accuracyList)) 
				accStdP1.append(np.std(accuracyList))
				specMeanP1.append(np.mean(specificityList))
				specStdP1.append(np.std(specificityList))
				sensMeanP1.append(np.mean(sensitivityList))
				sensStdP1.append(np.std(sensitivityList))
			else:
				accMeanP2.append(np.mean(accuracyList)) 
				accStdP2.append(np.std(accuracyList))
				specMeanP2.append(np.mean(specificityList))
				specStdP2.append(np.std(specificityList))
				sensMeanP2.append(np.mean(sensitivityList))
				sensStdP2.append(np.std(sensitivityList))


	# Graph illustration
	fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
	ax = axs[0,0]
	ax.errorbar(kList, accMeanP2, yerr=accStdP2, ecolor='orange')
	ax.set_title('Accuracy')
	ax = axs[1,0]
	ax.errorbar(kList, specMeanP2, yerr=specStdP2, ecolor='orange')
	ax.set_title('Specificity')
	ax = axs[0,1]
	ax.errorbar(kList, sensMeanP2, yerr=sensStdP2, ecolor='orange')
	ax.set_title('Sensitivity')

	fig.text(0.5, 0.04, 'k-neighbors', ha='center', va='center')
	fig.text(0.06, 0.5, 'Performance - %', ha='center', va='center', rotation='vertical')

	fig.suptitle('Cross-Validation: p=2')

	plt.show()


def main():
	readNParse('bc.txt')

	splitIndex = int(math.ceil(len(allPts) * 0.8))
	x_train = allPts[:splitIndex]
	x_test = allPts[splitIndex:]

	accuracyCalc(knn_classifier(x_test, x_train, x_test, 1, 2), True)

	crossValidation(allPts, 'kNN')

if __name__ == "__main__":
	main()

# REFERENCES
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://study.com/academy/lesson/pearson-correlation-coefficient-formula-example-significance.html
# https://www.youtube.com/watch?v=pkXonwx4DuY
# https://matplotlib.org/gallery/subplots_axes_and_figures/subplots_adjust.html#sphx-glr-gallery-subplots-axes-and-figures-subplots-adjust-py
# https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
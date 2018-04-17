# Brennan Hall - 861198641
# CS 171 18S
# Assignment 1

import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
from pandas.plotting import scatter_matrix

# classID: Which class of iris to be graphed
	# 1 for wine 1
	# 2 for wine 2
	# 3 for wine 3
	# 4 for iris-sesota
	# 5 for iris-versicolor
	# 6 for iris-virginica
# opID: Which operation the program should take
	# 0 for histogram
	# 1 for boxplot
	# 2 for correlogram
	# 3 for scatterplot
	# 4 for distance heatmap
# attrID: Which attribute to be graphed on the histogram
# attr2ID: Which second attribute to be compared to on the scatterplot
# bunNumID: How many bins to plot on the histogram
# uniP: Which value of p to use during distance calculation
# bigPrint: Set to True if all possible attribute*class*bin combinations for the given
# data set are to be drawn 


#  ----- CHANGE THESE NUMBERS FOR DIFFERENT INPUT TO FUNCTIONS -----
bigPrint = True
classID = 4
attrID = 1
attr2ID = 2
opID = 2
binNumID = 10
uniP = 2

# Global Lists
irisAttrGroups = [] 
wineAttrGroups = []
iPts = []
wPts = []
iNames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
wNames = ['NOPO', 'alcohol', 'malic-acid', 'ash', 'alcalinity-of-ash', 'magnesium', 'total-phenols', 'flavanoids', 'nonflavanoid-phenols', 'proanthocyanis', 'color-intensity', 'hue', 'OD280/OD315-of-diluted-wines', 'proline']

# ----- QUESTION 0 -----

# Reads and parses both iris and wine data into meaningful lists
def readNParse(irisFile, wineFile):
	global irisAttrGroups
	global wineAttrGroups
	global iPts
	global wPts

	# Split the data into line chunks
	with open(irisFile, 'r') as iData:
		iChunks = iData.read().splitlines()
	with open(wineFile, 'r') as wData:
		wChunks = wData.read().splitlines()

	# Parse iris data chunks into data points
	for m in range(len(iChunks)):
		temp = []
		splitChunk = iChunks[m].split(',')
		for q in range(5):
			if q == 4:
				temp.append(splitChunk[q])
			else:
				temp.append(float(splitChunk[q]))
		iPts.append(temp)

	# Parse wine data chunks into data points
	for r in range(len(wChunks)):
		temp = []
		splitChunk = wChunks[r].split(',')
		for c in range(len(wNames)):
			temp.append(float(splitChunk[c]))
		wPts.append(temp)

	# Parse iris data chunks into attribute groups
	for i in range(len(iNames)):
		temp = []
		for j in range(len(iChunks)):
			if iNames[i] == 'class':
				temp.append(iChunks[j].split(',')[i])
			else:
				temp.append(float(iChunks[j].split(',')[i]))
		irisAttrGroups.append(temp)

	# Parse wine data chunks into attribute groups
	for k in range(len(wNames)):
		temp = []
		for l in range(len(wChunks)):
			temp.append(float(wChunks[l].split(',')[k]))
		wineAttrGroups.append(temp)

# ----- QUESTION 1 -----

# Builds a histogram for a desired attribute
def buildHist(attrGroup, attrID, classID, binCount):
	bins = []
	floors = []

	# Title graph properly and isolate relevant data
	if classID == 1:
		plt.title(wNames[attrID] + ' - Wine Class 1')
		attrGroup = attrGroup[:59]
	elif classID == 2:
		plt.title(wNames[attrID] + ' - Wine Class 2')
		attrGroup = attrGroup[59:130]
	elif classID == 3:
		plt.title(wNames[attrID] + ' - Wine Class 3')
		attrGroup = attrGroup[130:]
	elif classID == 4:
		plt.title(iNames[attrID] + ' - Iris-sesota')
		attrGroup = attrGroup[:50]
	elif classID == 5:
		plt.title(iNames[attrID] + ' - Iris-versicolor')
		attrGroup = attrGroup[50:100]
	elif classID == 6:
		plt.title(iNames[attrID] + ' - Iris-virginica')
		attrGroup = attrGroup[100:]
	else:
		plt.title(iNames[attrID])

	# Calculate size and span of bins
	dataMax = max(attrGroup)
	dataMin = min(attrGroup)
	binSpan = (dataMax - dataMin)/binCount
	tempFloor = dataMin
	for i in range(binCount):
		bins.append(0)
		floors.append(tempFloor)
		tempFloor += binSpan
	floors.append(dataMax)

	# Fill bins
	for j in attrGroup:
		for k in range(binCount):
			if j >= floors[k] and j < floors[k+1]:
				bins[k] += 1
				break
			if j == dataMax:
				bins[binCount-1] += 1
				break

	# Tidy up floor values to make histogram more readable
	if binCount < 50:
		for f in range(len(floors)-1):
			floors[f] = str(floors[f]) + '-' + str(floors[f+1])
	floors = floors[:-1]

	print floors
	print bins

	# Aesthetics of histogram
	plt.bar(floors, bins, binSpan, color='lime')
	if binCount < 50:
		plt.xticks(floors)
		ax = plt.gca()
		plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

# ----- QUESTION 1 ~ PART 2 -----

def boxPlot(attrGroup, attrID, classID):
	if classID >= 4:
		plt.boxplot([attrGroup[:50], attrGroup[50:100], attrGroup[100:]])
		plt.xticks([1, 2, 3], ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
		plt.title(iNames[attrID])
	else:
		plt.boxplot([attrGroup[:59], attrGroup[59:130], attrGroup[130:]])
		plt.xticks([1, 2, 3], ['Wine Class 1', 'Wine Class 2', 'Wine Class 3'])
		plt.title(wNames[attrID])



# ----- QUESTION 2 ~ PART 1 -----

# Computes the Pearson correlation coefficient between two given pairs of features
def correlation(x, y):
	# Check if parameters are of equal length
	if len(x) != len(y):
		print 'correlation error: x and y are not of equal length'
		exit(1)

	# Compute coefficient
	return sum((x-np.mean(x))*(y-np.mean(y))) / (len(x)*np.std(x)*np.std(y))

# Builds the a matrix that contains the correlation coefficient 
# for every pair of features in the given dataset
def corrMatBuilder(attrSet, classID):
	# Adjust loop range for given dataset
	attrLen = len(attrSet)
	if classID >= 4:
		attrLen += -1

	# Fill matrix with dead value
	featureCorrMat = np.full((attrLen, attrLen), -2.0, dtype=float)

	# For every pair, compute the coefficient for every non-trivial pair
	# if it has not already been done and place it in the matrix
	for i in range(attrLen):
		for j in range(attrLen):
			if i == j:
				featureCorrMat[i][j] = 1
			if featureCorrMat[i][j] != -2.0:
				continue
			else:
				temp = correlation(attrSet[i], attrSet[j])
				featureCorrMat[i][j] = temp
				featureCorrMat[j][i] = temp
	return featureCorrMat

# Draws a heatmap from the given correlation matrix
def corrHeatMapper(corrMat, classID):
	plt.imshow(corrMat, cmap='RdYlGn', interpolation='nearest')
	if classID < 4:
		plt.title('Wine Attribute Correlations')
		plt.xticks(range(len(wNames-1)), wNames[-1:])
		plt.yticks(range(len(wNames-1)), wNames[-1:])
	else:
		plt.title('Iris Attribute Correlations')
		plt.xticks(range(len(iNames)-1), iNames[:-1])
		plt.yticks(range(len(iNames)-1), iNames[:-1])
	for i in range(len(corrMat)):
		corrMat[i] = np.around(corrMat[i], decimals=3)
	for i in range(len(corrMat)):
		for j in range(len(corrMat)):
			text = plt.text(j, i, corrMat[i, j], ha="center", va="center", color="w")
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	cax = plt.axes([0.85, 0.1, 0.075, 0.8])
	plt.colorbar(cax=cax)

# ----- QUESTION 2 ~ PART 2 -----

# Draws a scatterplot from the given pairs of attributes
def scPlot(x, y):
	plt.scatter(x[:50], y[:50], s=14.0, marker='o',c='indigo')
	plt.scatter(x[50:100], y[50:100], s=14.0, marker='o',c='purple')
	plt.scatter(x[100:], y[100:], s=14.0, marker='o',c='magenta')

	plt.title(iNames[attrID] + ' vs. ' + iNames[attr2ID])
	plt.xlabel(iNames[attrID])
	plt.ylabel(iNames[attr2ID])

# ----- QUESTION 2 ~ PART 3 -----

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

# Builds a matrix that contains the distance between every point in the
# given dataset
def distMatBuilder(dataPts, p):
	dataLen = len(dataPts)
	distMat = np.full((dataLen, dataLen), -2.0, dtype=float)

	# For every non-trivial pair of points, compute the distance
	# and place it in the matrix
	for i in range(dataLen):
		for j in range(dataLen):
			if i == j:
				distMat[i][j] = 0.0
			if distMat[i][j] == -2.0:
				tmp = distance(dataPts[i], dataPts[j], p)
				distMat[i][j] = tmp
				distMat[j][i] = tmp
	return distMat
# Draws a heatmap from the given distance matrix
def distHeatMapper(distMat, classID):
	plt.imshow(distMat, interpolation='nearest')
	if classID == 0:
		plt.title('Wine Data Point Distances (p = ' + str(uniP) + ')')
	else:
		plt.title('Iris Data Point Distances (p = ' + str(uniP) + ')')

	cax = plt.axes([0.85, 0.1, 0.075, 0.8])
	plt.colorbar(cax=cax)

# Finds the nearest data point to each data point in the given set of data
# and prints it to the console
def nearestPt(distMat, classID):
	dataLen = len(distMat)

	for i in range(dataLen):
		iMin = 5000
		iIndex = 0
		for j in range(dataLen):
			if distMat[i][j] != 0.0 and distMat[i][j] < iMin:
				iMin = distMat[i][j]
				iIndex = j
		if classID < 4:
			print ('Wine ' + str(i) + '\'s Nearest Wine: Wine ' + str(iIndex))
		else:
			print ('Iris ' + str(i) + '\'s (Class: ' + iPts[i][4] + ') Nearest Iris: Iris ' + str(iIndex) + ' (Class: ' + iPts[iIndex][4] + ')')


def main():
	readNParse('iris.txt', 'wine.txt')
	mainAttr = []
	mainAttrGroups = []
	mainPts = []
	if classID < 4:
		mainAttr = wineAttrGroups[attrID]
		mainAttrGroups = wineAttrGroups
		mainPts = wPts
	else:
		mainAttr = irisAttrGroups[attrID]
		mainAttrGroups = irisAttrGroups
		mainPts = iPts

	if opID == 0:
		if bigPrint:
			classList = [1, 2, 3]
			attrList = [1, 2, 3]
			binList = [5, 10, 50, 100]
			for curAttr in attrList:
				for curClass in classList:
					for curBin in binList:
						buildHist(wineAttrGroups[curAttr], curAttr, curClass, curBin)
						plt.show()
						exit(0)
		else:
			buildHist(mainAttr, attrID, classID, binNumID)
	elif opID == 1:
		if bigPrint:
			attrPath = [1, 2, 3]
			for curAttr in attrPath:
				boxPlot(wineAttrGroups[curAttr], curAttr, 0)
				plt.show()
			exit(0)
	elif opID == 2:
		corrMat = corrMatBuilder(mainAttrGroups, classID)
		print corrMat
		corrHeatMapper(corrMat, classID)
	elif opID == 3:
		scPlot(irisAttrGroups[attrID], irisAttrGroups[attr2ID])
	else:
		distMat = []
		if classID != 0:
			tempPts = []
			for z in mainPts:
				tempPts.append(z[:-1])
			distMat = distMatBuilder(tempPts, uniP)
		else:
			distMat = distMatBuilder(mainPts, uniP)
		print distMat
		nearestPt(distMat, classID)
		distHeatMapper(distMat, classID)

	plt.show()





if __name__ == "__main__":
	main()


# References:
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://study.com/academy/lesson/pearson-correlation-coefficient-formula-example-significance.html
# https://www.youtube.com/watch?v=pkXonwx4DuY
# https://matplotlib.org/gallery/subplots_axes_and_figures/subplots_adjust.html#sphx-glr-gallery-subplots-axes-and-figures-subplots-adjust-py
# https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
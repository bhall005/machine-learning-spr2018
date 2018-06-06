# Brennan Hall - 861198641
# CS 171 18S
# Assignment 3

import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
import copy
from pandas.plotting import scatter_matrix
from random import sample, seed
from numpy.random import choice

#  ----- CHANGE THESE NUMBERS FOR DIFFERENT INPUT TO FUNCTIONS -----

# Global Lists
iPts = []
iNames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']


# ----- QUESTION 0 -----

# Reads and parses iris data into meaningful lists
def readNParse(irisFile):
	global iPts

	# Split the data into line chunks
	with open(irisFile, 'r') as iData:
		iChunks = iData.read().splitlines()

	# Parse iris data chunks into data points
	for m in range(len(iChunks)):
		temp = []
		splitChunk = iChunks[m].split(',')
		for q in range(5):
			if q != 4:
				temp.append(float(splitChunk[q]))
		iPts.append(temp)

# ----- QUESTION 1 -----

# Computes the Euclidean distance from the given sets of data points
def distance(x, y):
  tmp = 0
  for i in range(len(x)):
    tmp += np.square(abs(float(x[i]) - float(y[i])))
  return tmp

# Determines the closest centroid to a point x
# Returns the index of the closest centroid
def closest_centroid(x, centroids):
	minDist = float('inf')
	minDex = 0
	for i in range(len(centroids)):
		tmpDist = distance(x, centroids[i])
		if tmpDist < minDist:
			minDist = tmpDist
			minDex = i
	return minDex

# Calculates the sum of sqaured errors for all data points
# and their respective centroids
def sse(x_input, cluster_assignments, cluster_centroids):
	tmp = 0
	# Calculate the sum of all data points to its assigned centroid squared
	for i in range(len(x_input)):
		tmp += np.square(distance(x_input[i], cluster_centroids[cluster_assignments[i]]))
	return tmp


def k_means_cs171(x_input, K, init_centroids):
	cluster_centroids = copy.deepcopy(init_centroids)
	cluster_assignments = []
	kClusters = [[] for x in range(K)]

	while True:
		# For each point x, find the nearest centroid and assign it to a cluster
		for i in x_input:
			assnIndex = closest_centroid(i, cluster_centroids)
			kClusters[assnIndex].append(i)
			cluster_assignments.append(assnIndex)

		# For each cluster j, make each new centroid the mean of all points in j
		new_centroids = []
		for cluster in kClusters:
			new_centroid = []
			for attrIndex in range(len(iNames)):
				attrSum = 0
				for point in cluster:
					attrSum += point[attrIndex]
				if len(cluster) != 0:
					attrMean = float(attrSum) / float(len(cluster))
				else:
					attrMean = 0
				new_centroid.append(attrMean)
			new_centroids.append(new_centroid)

		# If the centroids have ceased to shift, end the algorithm
		if new_centroids == cluster_centroids:
			break
		# Otherwise, rerun the algorithm using the new centroids
		cluster_centroids = copy.deepcopy(new_centroids)
		cluster_assignments = []
		kClusters = [[] for x in range(K)]


	return [cluster_assignments, cluster_centroids]

# ----- QUESTION 2 PART 1 -----

# Draws a simple line graph using given x and y values
def kneePlot(x_vals, y_vals, kList):
	plt.plot(x_vals, y_vals)
	plt.xticks(kList)
	plt.xlabel('K-Value')
	plt.ylabel('Sum of Squared Errors')

	plt.show()

# Runs kmeans for values of k from 1 to 10
# Graphs the knee plot of error for this data
def multiple_k_means(iPts):
	y_vals = []
	kList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

	# Prepare 10 different centroids, and only use amount needed
	# for the current K-value
	init_centroids = sample(iPts, 10)

	for k in kList:
		cluster_assignments, cluster_centroids = k_means_cs171(iPts, k, init_centroids[:k])
		y_vals.append(sse(iPts, cluster_assignments, cluster_centroids))

	kneePlot(kList, y_vals, kList)

# ----- QUESTION 2 PART 2 -----

# Draws a simple line graph using given x and y values
# Also draws errorbars given stddev of each y value
def kneePlotWithErrorBars(x_vals, y_vals, y_error, kList, max_iter, plusPlus):
	plt.errorbar(x_vals, y_vals, yerr=y_error, color='orange')
	plt.xticks(kList)
	plt.xlabel('K-Value')
	plt.ylabel('Mean of Sum of Squared Errors')
	if plusPlus:
		plt.title('K-Means++ for ' + str(max_iter) + ' Iterations')
	else:
		plt.title('K-Means for ' + str(max_iter) + ' Iterations')

	plt.show()

# Evaluates the sensitivity of the kmeans function
# 1. Run kmeans max_iter times
# 2. For each k, take the mean and stddev of each set of SSE's
# 3. Graph the resulting means for each k value
def kmeans_sensitivity_analysis(iPts):
	kList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	iterList = [2, 10, 100]

	for max_iter in iterList:
		y_vals = []
		y_error = []
		for k in kList:
			loopCnt = 0
			iter_y_vals = []
			while loopCnt < max_iter:
				init_centroids = sample(iPts, k)

				cluster_assignments, cluster_centroids = k_means_cs171(iPts, k, init_centroids)
				iter_y_vals.append(sse(iPts, cluster_assignments, cluster_centroids))

				loopCnt += 1
			y_vals.append(np.mean(iter_y_vals))
			y_error.append(np.std(iter_y_vals))

		kneePlotWithErrorBars(kList, y_vals, y_error, kList, max_iter, False)

# ----- QUESTION 3 PART 1 -----

# Runs kmeans on iPts for the given value of K but
# initializes centroids using the kmeans++ algorithm
def kmeans_plusplus(iPts, K):
	centers = []
	chosen_center_indexes = []
	loopCnt = 0

	# Choose one center uniformly at random
	init_center_index = choice(range(0, 149))
	chosen_center_indexes.append(init_center_index)
	centers.append(iPts[init_center_index])

	while loopCnt < K-1:
		dSquaredList = []

		# For each data point, compute the distance between it and
		# the nearest center
		for x in iPts:
			center_distances = []
			for c in centers:
				center_distances.append(distance(x, c))
			dSquaredList.append(np.square(min(center_distances)))

		# Choose one new center randomly using a weighted probablility
		# distribution where P(X) is proportional to its dSquared
		probs = dSquaredList/sum(dSquaredList)
		while True:
			new_center_index = choice(range(0, 150), 1, p=probs)
			if new_center_index in chosen_center_indexes:
				continue
			else:
				chosen_center_indexes.append(new_center_index)
				centers.append(iPts[new_center_index[0]])
				break
		loopCnt += 1

	# Once K centers have been chosen, run kmeans using the centers
	# as init_centroids
	cluster_assignments, cluster_centroids = k_means_cs171(iPts, K, centers)
	return [cluster_assignments, cluster_centroids]

# ----- QUESTION 3 PART 2 -----

# Evaluates the sensitivity of the kmeans++ function
# 1. Run kmeans++ max_iter times
# 2. For each k, take the mean and stddev of each set of SSE's
# 3. Graph the resulting means for each k value
def kmeans_plusplus_sensitivity_analysis(iPts):
	kList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	iterList = [2, 10, 100]

	for max_iter in iterList:
		y_vals = []
		y_error = []
		for k in kList:
			print k
			loopCnt = 0
			iter_y_vals = []
			while loopCnt < max_iter:
				init_centroids = sample(iPts, k)

				cluster_assignments, cluster_centroids = kmeans_plusplus(iPts, k)
				iter_y_vals.append(sse(iPts, cluster_assignments, cluster_centroids))

				loopCnt += 1
			y_vals.append(np.mean(iter_y_vals))
			y_error.append(np.std(iter_y_vals))

		kneePlotWithErrorBars(kList, y_vals, y_error, kList, max_iter, True)

def main():
	readNParse('iris.txt')

	#seed(9)
	init_centroids = sample(iPts, 3)

	#cluster_assignments, cluster_centroids = k_means_cs171(iPts, 3, init_centroids)
	#print cluster_assignments
	#print cluster_centroids
	#print sse(iPts, cluster_assignments, cluster_centroids)

	#multiple_k_means(iPts)

	#kmeans_sensitivity_analysis(iPts)
	kmeans_plusplus_sensitivity_analysis(iPts)





if __name__ == "__main__":
	main()

# REFERENCES
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://matplotlib.org/gallery/subplots_axes_and_figures/subplots_adjust.html#sphx-glr-gallery-subplots-axes-and-figures-subplots-adjust-py
# https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
# https://en.wikipedia.org/wiki/K-means%2B%2B
# https://stackoverflow.com/questions/5466323/how-exactly-does-k-means-work?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
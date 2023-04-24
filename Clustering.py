# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 22:21:27 2023

@author: bergd
Purpose: Perform clustering.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

if __name__ == "__main__":

    #initializing
    stdev = 3; samples = 500; clusters = 4; np.random.seed(781)
    if '-h' in sys.argv or '--help' in sys.argv:
        print ("Usage: %s [-c -n]" % sys.argv[0])
        print
        sys.exit(1)
    if '-c' in sys.argv:
        p = sys.argv.index('-c')
        clusters = int(sys.argv[p+1])
    if '-n' in sys.argv:
        p = sys.argv.index('-n')
        samples = int(sys.argv[p+1])
    if '-std' in sys.argv:
        p = sys.argv.index('-std')
        stdev = int(sys.argv[p+1])

	# Generating random data
    X, y_true = make_blobs(n_samples = samples, centers = clusters, cluster_std = stdev, random_state = 42)
	
	# Calculating
    kmeans = KMeans(clusters, random_state = 42)
    gmm = GaussianMixture(clusters, random_state = 42, tol = 1e-4, init_params = 'random' )
    gk = GaussianMixture(clusters, random_state = 42, tol = 1e-4, init_params = 'kmeans' )

	# Fitting models 
    k_fit = kmeans.fit_predict(X)
    g_fit = gmm.fit_predict(X)
    gk_fit = gk.fit_predict(X)
	
	# Plotting data
    plt.scatter(X[:, 0], X[:, 1], c = k_fit, s = 15, cmap = 'viridis')
    plt.title(f'KMeans (True), {clusters} clusters, {samples} samples')
    plt.savefig('figure1.png')
    plt.show()
    plt.scatter(X[:, 0], X[:, 1], c = g_fit, s = 15, cmap = 'viridis')
    plt.title(f'Gaussian Mixture Model, {clusters} clusters, {samples} samples')
    plt.savefig('figure2.png')
    plt.show()
    plt.scatter(X[:, 0], X[:, 1], c = gk_fit, s = 15, cmap = 'viridis')
    plt.title(f'Gaussian Mixture Model using KMeans, {clusters} clusters, {samples} samples')
    plt.savefig('figure3.png')
    plt.show()

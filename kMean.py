import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
import numpy as np

class kMean(object):

    def __init__(self,dataset,*,K=3):
        self.dataset = dataset
        self.features =self.dataset.columns
        self.K = K
        self.centroids = self.dataset.sample(n=K).values


    def get_closest_centroid(self,data_point):
        diff = self.centroids - data_point
        square_diff = np.sqrt( diff ** 2 )
        euclidean_dist = np.sum(square_diff, axis=1) / len(data_point)
        index_min_element = np.where(euclidean_dist == np.amin(euclidean_dist))
        return index_min_element[0][0]

    def assign_centroid(self):
        data = self.dataset[self.features].values
        self.dataset["centroid"] = list(map(self.get_closest_centroid, data))


    def calculate_mean_centroid(self):
        new_centroids = np.zeros(self.centroids.shape)
        for centroid_id in range(len(self.centroids)):
            closest_points = self.dataset["centroid"] == centroid_id
            data_points = self.dataset.loc[closest_points.values]
            data_points = data_points[data_points.columns[:-1]]
            new_centroids[centroid_id] = np.mean(data_points.values,axis=0)
        return new_centroids

    def termination_condition(self,new_centroids):
        epsilon = 0.001
        abs_diff = np.abs(new_centroids - self.centroids)
        res = np.all(abs_diff < epsilon)
        return res

    def Run(self):
        while True:
            self.assign_centroid()
            new_centroids = self.calculate_mean_centroid()
            if self.termination_condition(new_centroids):
                break
            self.centroids = new_centroids

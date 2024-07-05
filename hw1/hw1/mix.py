#!/usr/bin/env python
import numpy as np
from sklearn.cluster import KMeans


class GaussianMixture:
    def __init__(self, data, n_components=5):
        data = data.reshape((-1, data.shape[-1]))
        self.n_components = n_components
        self.n_features = data.shape[1]

        self.weights = np.zeros(self.n_components)
        self.means = np.zeros((self.n_components, self.n_features))
        self.covariances = np.zeros((self.n_components, self.n_features, self.n_features))
        self.kmeans = KMeans(n_clusters=self.n_components, n_init=1)
        self.kmeans.fit(data)
        self.update(data)

    def calc_prob(self, data):
        data = data.reshape((-1, data.shape[-1]))
        prob = []
        for comp in range(self.n_components):
            score = np.zeros(data.shape[0])
            if self.weights[comp] > 0:
                diff = data - self.means[comp]
                mult = np.einsum('ij,ij->i', diff, np.dot(np.linalg.inv(self.covariances[comp]), diff.T).T)
                score = (np.exp(-.5 * mult) / np.sqrt(2 * np.pi)) / np.sqrt(np.linalg.det(self.covariances[comp]))
            prob.append(score)

        return np.dot(self.weights, prob)

    def update(self, data):
        data = data.reshape((-1, data.shape[-1]))
        labels = self.kmeans.fit(data).labels_
        comp_data = np.zeros(self.n_components)
        self.weights = np.zeros(self.n_components)

        components, count = np.unique(labels, return_counts=True)
        comp_data[components] = count

        for comp in components:
            n = comp_data[comp]

            self.weights[comp] = n / np.sum(comp_data)
            self.means[comp] = np.mean(data[comp == labels], axis=0)
            self.covariances[comp] = 0 if comp_data[comp] <= 1 else np.cov(data[comp == labels].T)
            det = np.linalg.det(self.covariances[comp])
            if det <= 0:  # prevent 0 values
                self.covariances[comp] += np.eye(self.n_features) * 0.01

import numpy as np
from sklearn.cluster import KMeans


class GaussianMixture:
    def __init__(self, data, n_components=5):
        data = data.reshape((-1, data.shape[-1]))
        self.n_components = n_components

        self.weights = np.zeros(self.n_components)
        self.means = np.zeros((self.n_components, 3))
        self.covariances = np.zeros((self.n_components, 3,3))
        self.k_init(data)
        self.covariances_inverse = np.linalg.inv(self.covariances)
        self.covariances_determinant = np.linalg.det(self.covariances)

    def k_init(self, data):
        data = data.reshape((-1, data.shape[-1]))
        labels = KMeans(n_clusters=self.n_components, n_init=1).fit(data).labels_
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
                self.covariances[comp] += np.eye(3) * 0.001

    def calc_prob(self, data):
        data = data.reshape((-1, data.shape[-1]))
        prob = []
        for comp in range(self.n_components):
            score = np.zeros(data.shape[0])
            if self.weights[comp] > 0:
                diff = data - self.means[comp]
                mult = np.einsum('ij,ij->i', diff, np.dot(self.covariances_inverse[comp], diff.T).T)
                score = (np.exp(-.5 * mult) / np.sqrt(2 * np.pi)) / np.sqrt(self.covariances_determinant[comp])
            prob.append(score)

        return np.dot(self.weights, prob)

    def calc_N(self, xi, mu, covariance_inverse, covariance_determinant):
        xi_minus_mu = np.array(xi) - np.array(mu)
        xi_dimension = len(xi)
        numerator = np.e ** (-0.5 * np.dot(np.dot(xi_minus_mu.T, covariance_inverse), xi_minus_mu))
        denominator = ((2 * np.pi) ** (xi_dimension / 2)) * (covariance_determinant ** 0.5)
        return numerator / denominator

    # ric
    def evaluate_responsibility_for_pixel(self, pixel: list):
        probability = np.zeros(self.n_components)
        for cluster_index in range(self.n_components):
            covariance_inverse = self.covariances_inverse[cluster_index]
            covariance_determinant = self.covariances_determinant[cluster_index]
            N = self.calc_N(pixel, self.means[cluster_index], covariance_inverse, covariance_determinant)
            weight = self.weights[cluster_index]
            probability[cluster_index] = weight * N
        # Normalize
        total = np.sum(probability)
        return probability / total if total > 0 else np.zeros(self.n_components)

    def re_estimate_gmms_parameters(self, responsibility, pixels, cluster_index):
        sum_ric = np.sum(responsibility[:, cluster_index])
        mu = np.sum(pixels * responsibility[:, cluster_index].reshape(-1, 1), axis=0) / sum_ric

        weight = sum_ric / len(pixels)

        diff = pixels - mu
        sigma = np.dot((responsibility[:, cluster_index].reshape(-1, 1) * diff).T, diff) / sum_ric

        return weight, mu, sigma

    def update(self, pixels):
        responsibilities = np.zeros((pixels.shape[0], 5))
        for pixel_index in range(pixels.shape[0]):
            responsibilities[pixel_index] = self.evaluate_responsibility_for_pixel(pixels[pixel_index])

        for cluster_index in range(self.n_components):
            weight, mu, sigma = self.re_estimate_gmms_parameters(responsibilities, pixels, cluster_index)
            self.weights[cluster_index] = weight
            self.means[cluster_index] = mu
            self.covariances[cluster_index] = sigma

        for i in range(self.n_components):
            if np.linalg.det(self.covariances[i]) == 0:
                self.covariances[i] += 1e-6 * np.eye(self.covariances[i].shape[0])
        self.covariances_inverse = np.array([np.linalg.inv(self.covariances[i]) for i in range(self.n_components)])
        self.covariances_determinant = np.array([np.linalg.det(self.covariances[i]) for i in range(self.n_components)])


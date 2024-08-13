import numpy as np

from generator.ecg_generator import EcgGenerator

class EcgGeneratorMultidimensional(EcgGenerator):

    def __init__(self, autoencoder_params, n_samples=1):
        self.n_samples = n_samples
        super().__init__(autoencoder_params=autoencoder_params)

    def fit_to_normal_distribution(self, encoded_signals):
        mean_vector = np.mean(encoded_signals, axis=0)
        covariance_matrix = np.cov(encoded_signals, rowvar=False)
        return {
            'mean': mean_vector,
            'covariance': covariance_matrix
        }

    def sample_normal_distribution(self, distribution):
        mean_vector = distribution['mean']
        covariance_matrix = distribution['covariance']
        latent_samples = np.random.multivariate_normal(mean_vector, covariance_matrix, self.n_samples)
        return latent_samples
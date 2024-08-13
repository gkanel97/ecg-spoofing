import numpy as np
from sklearn.mixture import GaussianMixture

from generator.ecg_generator import EcgGenerator

class EcgGeneratorMixture(EcgGenerator):

    def __init__(self, autoencoder_params, n_samples=1, n_components=2, random_state=None):
        self.n_samples = n_samples
        self.n_components = n_components
        self.random_state = random_state
        super().__init__(autoencoder_params=autoencoder_params)

    def fit_to_normal_distribution(self, encoded_signals):
        n_features = encoded_signals.shape[1]
        gmm_models = []
        for i in range(n_features):
            latent_feature = encoded_signals[:, i].reshape(-1, 1)
            gmm = GaussianMixture(
                n_components=self.n_components, 
                random_state=self.random_state
            )
            gmm.fit(latent_feature)
            gmm_models.append(gmm)
        return gmm_models
    
    def sample_normal_distribution(self, gmm_models):
        latent_samples = []
        for gmm in gmm_models:
            feature_samples = gmm.sample(self.n_samples)[0]
            latent_samples.append(feature_samples)
        return np.concatenate(latent_samples, axis=1)
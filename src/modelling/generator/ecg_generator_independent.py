import numpy as np
from scipy.stats import norm

from src.modelling.generator.ecg_generator import EcgGenerator

class EcgGeneratorIndependent(EcgGenerator):

    def __init__(self, model_path, n_samples=1, data_path=EcgGenerator._DEFAULT_DATA_PATH):
        super().__init__(model_path=model_path, data_path=data_path)
        self.n_samples = n_samples

    def fit_to_normal_distribution(self, encoded_signals):
        n_features = encoded_signals.shape[1]
        mean_arr, std_arr = [], []
        for i in range(n_features):
            latent_feature = encoded_signals[:, i]
            mu, std = norm.fit(latent_feature)
            mean_arr.append(mu)
            std_arr.append(std)
        return {
            'mean': mean_arr,
            'std': std_arr
        }
    
    def sample_normal_distribution(self, distribution):
        mean_arr = distribution['mean']
        std_arr = distribution['std']
        latent_samples = []
        n_features = len(mean_arr)
        for i in range(n_features):
            feature_samples = np.random.normal(mean_arr[i], std_arr[i], self.n_samples)
            latent_samples.append(feature_samples)
        return np.array(latent_samples).T
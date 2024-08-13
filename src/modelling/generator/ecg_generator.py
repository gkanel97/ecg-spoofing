
# import numpy as np
# from keras.models import Model
from tensorflow.keras.models import load_model

from abc import ABC, abstractmethod

# from utils.ecg_processing import EcgSignalProcessor
from src.dataset import EcgSignalProcessor

class EcgGenerator(ABC):

    # def __init__(self, autoencoder_params, templates_per_user=200):
    #     self.ecg_processor = EcgSignalProcessor(
    #         data_path='data/autonomic-aging-cardiovascular/1.0.0', 
    #         templates_to_extract=templates_per_user
    #     )

    #     autoencoder = load_model(autoencoder_params['filepath'])
    #     self.encoder = Model(
    #         inputs=autoencoder.input, 
    #         outputs=autoencoder.get_layer(autoencoder_params['encoder_output_layer']).output
    #     )
    #     self.decoder = Model(
    #         inputs=autoencoder.get_layer(autoencoder_params['decoder_input_layer']).input,
    #         outputs=autoencoder.get_layer(autoencoder_params['decoder_input_layer']).output
    #     )

    _DEFAULT_DATA_PATH = '../../data/raw/autonomic-aging-cardiovascular/1.0.0'

    def __init__(self, model_path, data_path=_DEFAULT_DATA_PATH, templates_per_user=200):
        self.ecg_processor = EcgSignalProcessor(
            data_path=data_path, 
            templates_to_extract=templates_per_user
        )

        autoencoder = load_model(model_path)
        self.encoder = autoencoder.get_layer('encoder')
        self.decoder = autoencoder.get_layer('decoder')
    
    def load_and_encode_signals(self, user_id):
        templates = self.ecg_processor.read_user_ecg_signals(user_id)
        if templates is None or len(templates) == 0:
            return None
        encoded_signals = self.encoder.predict(templates, verbose=0)
        return encoded_signals
    
    @abstractmethod
    def sample_normal_distribution(self, distribution):
        pass

    @abstractmethod
    def fit_to_normal_distribution(self, encoded_signals):
        pass
    
    def decode_samples(self, latent_samples):
        return self.decoder.predict(latent_samples, verbose=0)

    def generate_synthetic_signals(self, user_id):
        encoded_signals = self.load_and_encode_signals(user_id)
        if encoded_signals is None:
            return None
        distribution = self.fit_to_normal_distribution(encoded_signals)
        latent_samples = self.sample_normal_distribution(distribution)
        synthetic_templates = self.decode_samples(latent_samples)
        return synthetic_templates
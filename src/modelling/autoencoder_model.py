from keras.models import Model, load_model
from keras.callbacks import EarlyStopping

from abc import ABC, abstractmethod

class AutoencoderModel(ABC):

    def __init__(self, encoder_output_layer, decoder_input_layer, model_name):
        self.encoder_output_layer = encoder_output_layer
        self.decoder_input_layer = decoder_input_layer
        self.model_name = model_name
        self.autoencoder = None

    @abstractmethod
    def build_model(self):
        pass

    def set_autoencoder(self, autoencoder):
        self.autoencoder = autoencoder

    def load_model(self, model_path):
        self.autoencoder = load_model(model_path)

    def get_encoder_model(self):
        encoder = Model(
            inputs=self.autoencoder.input,
            outputs=self.autoencoder.get_layer(self.encoder_output_layer).output
        )
        return encoder
    
    def get_decoder_model(self):
        decoder = Model(
            inputs=self.autoencoder.get_layer(self.decoder_input_layer).input,
            outputs=self.autoencoder.output
        )
        return decoder

    def train_model(self, x_train, x_val, max_epochs, batch_size, optimizer):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        self.autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
        history = self.autoencoder.fit(
            x_train, x_train,
            epochs=max_epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_val, x_val),
            callbacks=[early_stopping]
        )
        self.autoencoder.save(f'{self.model_name}.keras')
        return history
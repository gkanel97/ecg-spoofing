### ECG Spoofing with Autoencoders and Gaussian Mixture Models
This repository contains code written for my dissertation on ECG signal spoofing with autoencoders and Gaussian mixture models. The goal of this project is to investigate if autoencoders can create probabilistic models of users that are able to circumvent a biometric authentication system. First, the autoencoder architecture is optimised to extract features from ECG signals. Second, Gaussian mixture models are used to create a probabilistic model of users. Third, random samples from the probabilistic are used to generate synthetic ECG data. Fourth, synthetic data is evaluated against a biometric authentication system (presentation attack).

#### Methodology

##### Part 1: Data preparation
ECG signals are loaded from a database uing the src/dataset.py script. Then, signal are filtered and split into heartbeats. Relevant code is found on notebook signal_processing.ipynb.

##### Part 2: Autoencoder architecture optimisation
Notebooks are used to optimise the autoencoder architecture. The following parameters are optimised:
- Type of hidden layers (notebooks/conv/conv_vs_dense.ipynb)
- Activation function (notebooks/dense/training_activation.ipynb)
- Latent space size (notebooks/dense/latent_space_size.ipynb)
- Shape of hidden layers (notebook/dense/hidden_layers_shape.ipynb)

##### Part 3: Inference
This step includes sampling from probabilistic models and generating synthetic signals. Relevant code is found on notebook inference.ipynb and in the supporting classes defined in src/modelling/generator folder. The following mixture models are used to generate probabilistic models:
- Gaussian mixture model with 1 component
- Gaussian mixture model with 2 components
- Multivariate Gaussian mixture model

##### Part 4: Presentation attack
A presentation attack against a biometric authenticator is orchestrated using helper code from src/authentication.py and notebooks under the authentication folder.

#### How to run the code
Follow the next steps to run the code.
- Step 1: Create a virtual environment and install required libraries.
```
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
```
- Step 2: Create folders for models and data.
```
  mkdir models
  mkdir -p data/raw
  mkdir -p data/processed
```
- Step 3: Download ECG dataset, process signals and split in train, test and validation sets.
```
  wget -r -N -c -np -P data/raw/ https://physionet.org/files/autonomic-aging-cardiovascular/1.0.0/
  python dataset.py
```
- Step 4: Run the notebook code in any order.

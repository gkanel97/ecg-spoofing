import os
import wfdb
import pickle
import numpy as np
from tqdm import tqdm
from biosppy.signals import ecg
from sklearn.preprocessing import MinMaxScaler

class EcgSignalProcessor():

    def __init__(self, data_path, templates_to_extract=20):
        self.data_path = data_path
        self.templates_to_extract = templates_to_extract
        self.scaler = MinMaxScaler()

    def load_ecg_signal(self, file_name):
        try:
            file_path = os.path.join(self.data_path, file_name)
            record = wfdb.rdrecord(file_path)
            signal = record.p_signal[:, 0]
            fs = record.fs
        except Exception as e:
            print(f'Error processing {file_name}: {e}')
            signal, fs = None, None
        finally:
            return signal, fs

    def extract_and_scale_templates(self, signal, fs):
        try:
            ecg_dict = ecg.ecg(signal, sampling_rate=fs, show=False)
            templates = ecg_dict['templates'][:self.templates_to_extract]
            scaled_templates = MinMaxScaler().fit_transform(templates.T).T
        except Exception as e:
            print(f'Error processing signal: {e}')
            scaled_templates = []
        finally:
            return scaled_templates
        
    def read_ecg_dataset(self, start_idx=0, end_idx=500):
        data_files = [f for f in os.listdir(self.data_path) if f.endswith('.dat')]
        data = []
        labels = []
        pbar = tqdm(total=end_idx-start_idx)
        for file_name in data_files[start_idx:end_idx]:
            file_name_without_ext = file_name[:-4]
            signal, fs = self.load_ecg_signal(file_name_without_ext)
            if signal is not None:
                templates = self.extract_and_scale_templates(signal, fs)
                data.extend(templates)
                labels.extend([int(file_name_without_ext)]*len(templates))
            pbar.update(1)
        pbar.close()
        return np.array(data), labels
    
    def read_user_ecg_signals(self, user_id):
        file_name = f'{user_id:04d}'
        try:
            signal, fs = self.load_ecg_signal(file_name)
            if signal is not None:
                templates = self.extract_and_scale_templates(signal, fs)
                return templates
        except Exception as e:
            print(f'Error processing {file_name}: {e}')
            return None
        
if __name__ == '__main__':

    ecg_processor = EcgSignalProcessor(
    data_path='data/raw/autonomic-aging-cardiovascular/1.0.0', 
    templates_to_extract=100,
    )
        
    X_train, y_train = ecg_processor.read_ecg_dataset(start_idx=0, end_idx=600)
    X_val, y_val = ecg_processor.read_ecg_dataset(start_idx=600, end_idx=800)
    X_test, y_test = ecg_processor.read_ecg_dataset(start_idx=800, end_idx=1000)

    with open('data/processed/autoencoder_data.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test': X_test,
            'X_val': X_val,
            'y_train': y_train,
            'y_test': y_test,
            'y_val': y_val
        }, f)
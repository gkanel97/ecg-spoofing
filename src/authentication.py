import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class EcgAuthenticator:

    def __init__(self):
        self.auth_model = load_model('../../models/authentication/identification_cnn.keras')

    def identify(self, templates, majority_votes=1):
        y_pred = self.auth_model.predict(templates, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)
        if majority_votes > 1:
            ret_val = []
            for i in range(0, len(y_pred), majority_votes):
                majority_vote = np.argmax(np.bincount(y_pred[i:i+majority_votes]))
                ret_val.append(majority_vote)
        else:
            ret_val = y_pred
        return ret_val
    
    def evaluate_authenticator(self, y_pred, y_true):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
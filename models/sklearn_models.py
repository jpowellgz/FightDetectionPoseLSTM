
from typing import Any

import numpy as np
from fight_detection_pose_lstm.model_base import Model, ModelArgs
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC


class SKLearnModel(Model):
    def __init__(self, model_args: ModelArgs):
        super().__init__(model_args)
        self.init_model()

    def load_model(self):
        return
        
    def save_model(self):
        return
        
    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(x_train, y_train)

    def inference(self, x_input: np.ndarray) -> np.ndarray:
        pred = self.model.predict(x_input)
        return pred
    
    def evaluate(self, x_input:np.ndarray, y_input: np.ndarray) -> Any:
        score = self.model.score(x_input, y_input)
        return score

class SVCModel(SKLearnModel):
    def init_model(self):
        self.model = SVC()

class AdaBoostModel(SKLearnModel):
    def init_model(self):
        self.model = AdaBoostClassifier()

class ForestModel(SKLearnModel):
    def init_model(self):
        self.model = RandomForestClassifier()
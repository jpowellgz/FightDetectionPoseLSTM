from abc import abstractmethod, ABC
from dataclasses import dataclass
import numpy as np


@dataclass
class ModelArgs:
    local_path: str

class Model(ABC):
    """Base class to get a model, train it on vectors and perform inference"""
    def __init__(self, model_args: ModelArgs):
        self.args = model_args
    
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def inference(self, x_input: np.ndarray):
        pass

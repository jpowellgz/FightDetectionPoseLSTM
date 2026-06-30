from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
from fight_detection_pose_lstm.model_base import Model, ModelArgs


@dataclass
class LSTMArgs(ModelArgs):
    hidden_units: int
    epochs: int
    sequence_length: int
    vector_size: int
    save_path: str | None = None
    train_size: float = 0.8


class TensorflowLSTM(Model):
    def __init__(self, model_args: LSTMArgs):
        super().__init__(model_args)
        self.init_model()

    def init_model(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        self.args.hidden_units,
                        input_shape=(self.args.sequence_length, self.args.vector_size),
                    )
                ),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    def load_model(self):
        if self.args.model_path is not None:
            self.model.load_weights(self.args.model_path, skip_mismatch=False)
        

    def save_model(self):
        if self.args.save_path is not None:
            self.model.save_weights("self.args.save_path")
        

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(x_train, y_train,epochs=self.args.epochs)

    def inference(self, x_input: np.ndarray) -> np.ndarray:
        pred = self.model.predict(x_input, verbose=0)
        return pred
    
    def evaluate(self, x_input:np.ndarray, y_input: np.ndarray) -> Any:
        score = self.model.evaluate(x_input, y_input, verbose=0)
        return score

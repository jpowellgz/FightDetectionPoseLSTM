from dataclasses import dataclass

from fight_detection_pose_lstm.model_base import Model, ModelArgs


@dataclass
class LSTMArgs(ModelArgs):
    hidden_units: int


class TensorflowLSTM(Model):
    def __init__(self, model_args: LSTMArgs):
        self.hidden_units = model_args.hidden_units

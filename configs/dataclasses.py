from dataclasses import dataclass

@dataclass
class DirectoryConfig:
    directory: str
    fight_subdirectory: str
    no_fight_subdirectory: str

DIRECTORY_CONFIG = "directory_config"

@dataclass
class ExtractionConfig:
    num_frames: int

EXTRACTION_CONFIG = "extraction_config"

@dataclass
class AngleCalculatorConfig:
    angle_bins: int
    fight_pairs_indexes: list[int]

ANGLE_CONFIG = "angle_calculator"

KEYPOINT_CONFIG = "keypoint_model"
CLASSIF_MODEL = "classification_model"
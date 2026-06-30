from dataclasses import dataclass

@dataclass
class DirectoryConfig:
    directory: str
    fight_subdirectory: str
    no_fight_subdirectory: str
    output_path: str

@dataclass
class ExtractionConfig:
    num_frames: int


@dataclass
class AngleCalculatorConfig:
    angle_bins: int
    fight_pairs_indexes: list[int]


@dataclass(frozen=True)
class ConfigKeys:
    directory: str = "directory_config"
    extraction: str = "extraction_config"
    angle: str = "angle_calculator"
    keypoint: str = "keypoint_model"
    classification: str = "classification_model"

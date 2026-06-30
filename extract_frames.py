import argparse
import os
from typing import Any
from fight_detection_pose_lstm.frame_extraction import VideoFrameExtractor
from fight_detection_pose_lstm.config import DirectoryConfig, ExtractionConfig, ConfigKeys
from fight_detection_pose_lstm.utils import check_path, load_config


""" Extract frames from videos based on the given config file. Default configs/extract_config.json"""

def check_config(config: dict[str, Any]) -> None:
    if ConfigKeys.extraction not in config:
        raise AttributeError(f"Missing {ConfigKeys.extraction}")
    if ConfigKeys.directory not in config:
        raise AttributeError(f"Missing {ConfigKeys.directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="extract_frames", description="Extract frames from a fight video dataset")
    parser.add_argument("config", type=str, help="Path to the config")
    args = parser.parse_args()

    # Check config and directory paths
    check_path(args.config)
    config_dict = load_config(args.config)
    check_config(config_dict)
    extraction_config = ExtractionConfig(**config_dict[ConfigKeys.extraction])
    dir_config = DirectoryConfig(**config_dict[ConfigKeys.directory])

    check_path(dir_config.directory)
    fight_dir = os.path.join(dir_config.directory, dir_config.fight_subdirectory)
    no_fight_dir = os.path.join(dir_config.directory, dir_config.no_fight_subdirectory)
    check_path(fight_dir)
    check_path(no_fight_dir)

    # Extract frames from videos

    fight_videos = [filename for filename in os.listdir(fight_dir) if filename.endswith(".mp4") or filename.endswith(".avi")]
    no_fight_videos = [filename for filename in os.listdir(no_fight_dir) if filename.endswith(".mp4") or filename.endswith(".avi")]

    for video in fight_videos:
        extractor = VideoFrameExtractor(
            video_path=os.path.join(dir_config.directory, dir_config.fight_subdirectory, video),
            output_directory=os.path.join(dir_config.output_path, dir_config.fight_subdirectory, video.split(".")[0]),
            num_frames=extraction_config.num_frames,
        )
        extractor.extract_frames()
    
    for video in no_fight_videos:
        extractor = VideoFrameExtractor(
            video_path=os.path.join(dir_config.directory, dir_config.no_fight_subdirectory, video),
            output_directory=os.path.join(dir_config.output_path, dir_config.no_fight_subdirectory, video.split(".")[0]),
            num_frames=extraction_config.num_frames,
        )
        extractor.extract_frames()

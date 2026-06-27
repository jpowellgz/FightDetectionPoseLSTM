import argparse
import os
import json
from fight_detection_pose_lstm.frame_extraction import VideoFrameExtractor
from configs.dataclasses import DirectoryConfig, ExtractionConfig, DIRECTORY_CONFIG, EXTRACTION_CONFIG
from utils import check_path, load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="extract_frames", description="Extract frames from a fight video dataset")
    parser.add_argument("config", type=str, help="Path to the config")

    args = parser.parse_args()
    check_path(args.config)
    config_dict = load_config(args.config)
    extraction_config = ExtractionConfig(**config_dict[EXTRACTION_CONFIG])
    dir_config = DirectoryConfig(**config_dict[DIRECTORY_CONFIG])


    check_path(dir_config.directory)
    fight_dir = os.path.join(dir_config.directory, dir_config.fight_subdirectory)
    no_fight_dir = os.path.join(dir_config.directory, dir_config.no_fight_subdirectory)
    check_path(fight_dir)
    check_path(no_fight_dir)

    fight_videos = [filename for filename in os.listdir(fight_dir) if filename.endswith(".mp4") or filename.endswith(".avi")]
    no_fight_videos = [filename for filename in os.listdir(no_fight_dir) if filename.endswith(".mp4") or filename.endswith(".avi")]

    FRAMES_DIR = "frames"
    for video in fight_videos:
        extractor = VideoFrameExtractor(
            video_path=os.path.join(dir_config.directory, dir_config.fight_subdirectory, video),
            output_directory=os.path.join(dir_config.directory, FRAMES_DIR, dir_config.fight_subdirectory, video.split(".")[0]),
            num_frames=extraction_config.num_frames,
        )
        extractor.extract_frames()
    
    for video in no_fight_videos:
        extractor = VideoFrameExtractor(
            video_path=os.path.join(dir_config.directory, dir_config.no_fight_subdirectory, video),
            output_directory=os.path.join(dir_config.directory, FRAMES_DIR, dir_config.no_fight_subdirectory, video.split(".")[0]),
            num_frames=extraction_config.num_frames,
        )
        extractor.extract_frames()

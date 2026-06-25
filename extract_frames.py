import argparse
import os
from fight_detection_pose_lstm.frame_extraction import VideoFrameExtractor

def check_dir(directory: str):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"{directory} not found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="extract_frames", description="Extract frames from a fight video dataset")
    parser.add_argument("directory", type=str, help="Path to the root directory of the dataset")
    parser.add_argument("fight_subdirectory", type=str, help="Name of the fight subdirectory")
    parser.add_argument("no_fight_subdirectory", type=str, help="Name of the no-fight subdirectory")
    parser.add_argument("num_frames", type=int, default=10, help="Number of frames to extract per video")

    args = parser.parse_args()

    check_dir(args.directory)
    fight_dir = os.path.join(args.directory, args.fight_subdirectory)
    no_fight_dir = os.path.join(args.directory, args.no_fight_subdirectory)
    check_dir(fight_dir)
    check_dir(no_fight_dir)

    fight_videos = [filename for filename in os.listdir(fight_dir) if filename.endswith(".mp4") or filename.endswith(".avi")]
    no_fight_videos = [filename for filename in os.listdir(no_fight_dir) if filename.endswith(".mp4") or filename.endswith(".avi")]

    FRAMES_DIR = "frames"
    for video in fight_videos:
        extractor = VideoFrameExtractor(
            video_path=os.path.join(args.directory, args.fight_subdirectory, video),
            output_directory=os.path.join(args.directory, FRAMES_DIR, args.fight_subdirectory, video.split(".")[0]),
            num_frames=args.num_frames,
        )
        extractor.extract_frames()
    
    for video in no_fight_videos:
        extractor = VideoFrameExtractor(
            video_path=os.path.join(args.directory, args.no_fight_subdirectory, video),
            output_directory=os.path.join(args.directory, FRAMES_DIR, args.no_fight_subdirectory, video.split(".")[0]),
            num_frames=args.num_frames,
        )
        extractor.extract_frames()

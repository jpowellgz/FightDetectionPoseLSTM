import argparse
from fight_detection_pose_lstm.frame_extraction import VideoFrameExtractor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("output_directory")
    parser.add_argument("num_frames")

    args = parser.parse_args()
    extractor = VideoFrameExtractor(
        video_path=args.video_path,
        output_directory=args.output_directory,
        num_frames=int(args.num_frames),
    )
    extractor.extract_frames()

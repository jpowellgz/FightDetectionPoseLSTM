import cv2
import tqdm
import numpy as np
import os
import math
from src.fightdetectionposelstm.logging import logger

class VideoReader:
    """Class to create an OpenCV video reader."""
    def __init__(self, video_path: str) -> None:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"{video_path} not found")
        self.vid_capture = cv2.VideoCapture(video_path)
        if not self.vid_capture.isOpened():
            logger.error("Error opening the video file")
        logger.info(f"Starting video reader with path {video_path}")

        self.fps = int(self.vid_capture.get(5))
        frame_width = int(self.vid_capture.get(3))
        frame_height = int(self.vid_capture.get(4))
        self.frame_size = (frame_width,frame_height)
        self.frames = int(self.vid_capture.get(7))
        self.ret = True
        self.frame = None

        self.stop = False
        self.pause = False
    
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read a frame"""
        if not self.stop and not self.pause:
            if not self.ret:
                self.stop_reading()
                return self.ret, None
            else:
                self.ret, self.frame = self.vid_capture.read()
                return self.ret, self.frame

    def stop_reading(self):
        """Stop reading frames and release"""
        logger.info("Stopping video reader")
        self.stop = True
        self.vid_capture.release()

    def pause_reading(self):
        """Stop reading but don't release yet"""
        self.pause = True
    
    def resume_reading(self):
        """Set the pause off"""
        self.pause = False

    def get_fps(self) -> int:
        return self.fps
    
    def get_frame_size(self) -> tuple[int]:
        return self.frame_size


class VideoWriter:
    """Class to create an OpenCV video writer."""
    def __init__(self, video_path, fps, frame_size):
        self.writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    
    def write(self, frame):
        self.writer.write(frame)
    
    def release(self):
        self.writer.release()



class VideoFrameExtractor(VideoReader):
    """Class to create an OpenCV video reader that takes frames and saves them to a path."""
    def __init__(self, video_path: str, output_directory: str, num_frames: int = 10) -> None:
        super().__init__(video_path)
        self.video_path = video_path
        self.output_directory = output_directory
        self.num_frames = num_frames
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)


    def get_frames_from_video(self) -> list[np.ndarray]:
        """Return k frames from a video, equally spaced

        Args:
            video_path (str): _description_
            function (callable): _description_
            k (int, optional): _description_. Defaults to 10.

        Returns:
            list[np.ndarray]: _description_
        """
        logger.info(f"Starting extraction from video {self.video_path}")
        test_frames = []
        every_k_frames = int(self.frames / self.num_frames)

        for i in tqdm.tqdm(range(self.num_frames), unit="frames"):
            frame_number = i * every_k_frames
            self.vid_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = self.read()
            if ret:
                test_frames.append(frame)   
            else:
                break
        self.stop_reading()
        if len(test_frames) == self.num_frames:
            logger.info("Frames extracted successfully")
        else:
            logger.warning("Frames not extracted successfully")
        return test_frames

    def save_frame(self, frame: np.ndarray, number: int, digits: int) -> None:
        """Save a frame

        Args:
            frame (np.ndarray): frame
            number (int): number id
            digits (int): number of digits to use
        """
        file_name = os.path.join(self.output_directory, f"frame_{number:0{digits}}.jpg")
        cv2.imwrite(filename=file_name, img=frame)

    def extract_frames(self) -> None:
        frames = self.get_frames_from_video()
        logger.info(f"Saving frames to {self.output_directory}")
        num_frames = len(frames)
        digits = math.ceil(math.log10(num_frames))
        for idx, frame in enumerate(frames):
            self.save_frame(frame, idx, digits)

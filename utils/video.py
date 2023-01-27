import subprocess
import json
from moviepy.tools import subprocess_call as moviepy_subprocess_call


def ffmpeg_cut_video(input_path: str, output_path: str, start_sec: float = None, end_sec: float = None,
                     video_codec: str = 'h264') -> None:
    """
    Cut a video and remove audio. This function is modified from: moviepy.video.io.ffmpeg_tools.ffmpeg_extract_subclip

    Args:
        input_path: input video path
        start_sec: start second
        end_sec: end second
        output_path: path to save new video
        video_codec: video codec, default is h264 because 'copy' codec may mess up the metadata
    """
    assert start_sec or end_sec, "No start/end time provided!"
    if start_sec is None:
        start_sec = 0

    cmd = ["ffmpeg", "-y", "-i", input_path]

    if start_sec:
        cmd += ["-ss", "%0.2f" % start_sec]
    if end_sec:
        cmd += ["-t", "%0.2f" % (end_sec - start_sec)]

    cmd += ["-map", "0", "-vcodec", video_codec, "-an", output_path]

    moviepy_subprocess_call(cmd)


def get_video_metadata(path: str) -> dict:
    """
    Get video metadata

    Args:
        path: path to video

    Returns:
        a dictionary with keys: length, fps, num_frames
    """
    result = subprocess.check_output(
        f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{path}"',
        shell=True).decode()
    fields = json.loads(result)['streams'][0]

    return {
        'length': float(fields['duration']),
        'fps': eval(fields['r_frame_rate']),
        'num_frames': int(fields['nb_frames'])
    }


# opencv-python == 4.5.5.64
# import numpy as np
# from typing import Tuple
# import cv2
# class BaseVideoWriter:
#     def write_frame(self, frame: np.ndarray):
#         """
#         Write next frame to the video
#
#         Args:
#             frame: a frame in the form of a numpy array
#         """
#         raise NotImplementedError()
#
#     def close(self):
#         """
#         Close the video writer
#         """
#         raise NotImplementedError()
#
#
# class OpenCVVideoWriter(BaseVideoWriter):
#     def __init__(self, path: str, resolution: Tuple[int, int], fps: float, video_format: str = 'MP4V'):
#         if video_format.upper() == 'MJPG':
#             ext = '.avi'
#         elif video_format.upper() == 'MP4V':
#             ext = '.mp4'
#         else:
#             raise ValueError('only support format MP4V and MJPG')
#
#         if not path.endswith(ext):
#             path += ext
#
#         self.resolution = resolution
#         self.reversed_resolution = resolution[::-1]
#         self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*video_format), fps, resolution)
#
#     def write_frame(self, frame: np.ndarray):
#         if frame.shape[:2] != self.reversed_resolution:
#             frame = cv2.resize(frame, self.resolution)
#         self.writer.write(frame)
#
#     def close(self):
#         self.writer.release()

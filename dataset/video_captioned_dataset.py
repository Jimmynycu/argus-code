from torch.utils.data import Dataset
import os
import random
import torch
from PIL import Image
import numpy as np
import glob
import cv2

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', 'MP4', 'AVI', 'MOV']

class VideoCaptionedDataset(Dataset):
    '''
    given a folder containing videos, this class will output a sequence of frames from a random video
    depending on the sample_frames and skip_frame parameters
    '''
    def __init__(self, base_folder_or_file,
                 caption_folder_or_file,
                 width=None, height=None, 
                 wanted_name_in_dataset=None,
                 sample_frames=25, dataset_size=None,
                 rotation=False,
                 frame_rate=None,
                 frame_interval=1,
                 fixed_start_frame=False,
                 clip_info_path=None,
                 dense_calibration=False,
                 calibration_img_size=None,
                 full_sampling=False, 
                 cached_motion_path: str = None,
                 is_train=True,
                 ):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
            dataset_size (int): Number of videos to use. If provided, will randomly select videos from the folder.
            fov_path (str): Path to the file containing the field of view information.
            full_sampling (bool): If True, sample sample_frames frames from the video, starting from the first frame to the last frame.
        """
    
        # Define the path to the folder containing video frames
        self.videos = []
        self.captions = []
        if type(base_folder_or_file) == str:
            if os.path.isfile(base_folder_or_file):
                self.videos = [base_folder_or_file]
                self.caption = [caption_folder_or_file]
            else:
                if clip_info_path is not None:
                    with open(clip_info_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            category, video_id, clip = line.strip().split('\t')
                            self.videos.append(os.path.join(base_folder_or_file, category, video_id, clip+'.mp4'))
                else:
                    for ext in VIDEO_EXTENSIONS:
                        self.videos.extend(glob.glob(os.path.join(base_folder_or_file, f'**/*{ext}'), recursive=True))
                for video in self.videos:
                    self.captions.append(video.replace(base_folder_or_file, caption_folder_or_file)[:-4]+'.txt')

        elif type(base_folder_or_file) == list:
            if clip_info_path is not None:
                assert len(base_folder_or_file) == 1, "Clip info path is only supported for a single folder"
                with open(clip_info_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        category, video_id, clip = line.strip().split('\t')
                        self.videos.append(os.path.join(base_folder_or_file[0], category, video_id, clip+'.mp4'))
                for video in self.videos:
                    self.captions.append(video.replace(base_folder_or_file[0], caption_folder_or_file[0])[:-4]+'.txt')
            else:
                for (base_folder, caption_folder) in zip(base_folder_or_file, caption_folder_or_file):
                    if os.path.isfile(base_folder) and any(base_folder.endswith(ext) for ext in VIDEO_EXTENSIONS):
                        self.videos.append(base_folder)
                        self.captions.append(caption_folder)
                    else:
                        for ext in VIDEO_EXTENSIONS:
                            self.videos.extend(glob.glob(os.path.join(base_folder, f'**/*{ext}'), recursive=True))
                            self.captions.extend(glob.glob(os.path.join(caption_folder, f'**/*{ext}'), recursive=True))

        # if wanted_name_in_dataset is not None: # wanted_name_in_dataset is a list of strings
        #     self.videos = [video for video in self.videos if any(name in video.lower() for name in wanted_name_in_dataset)]

        if dataset_size is not None:
            self.videos = self.videos * dataset_size
            self.captions = self.captions * dataset_size

        self.channels = 3
        self.width = width
        self.height = height
        self.calibration_img_size = calibration_img_size
        self.sample_frames = sample_frames

        self.rotation = rotation

        self.frame_rate = frame_rate
        self.fixed_start_frame = fixed_start_frame
        self.frame_interval = frame_interval
        self.full_sampling = full_sampling

        print("Number of videos in the dataset:", len(self.videos))

        self.dense_calibration = dense_calibration
        self.cached_motion = cached_motion_path is not None
        self.cached_motion_path = cached_motion_path

        self.is_train = is_train

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, height, width).
        """
        ret = {}

        video_path = self.videos[idx]
        caption_path = self.captions[idx]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        frames, frames_dense = [], []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if self.full_sampling:
            frame_interval = (total_frames - 1) // (self.sample_frames - 1)
        elif self.frame_rate is None:
            frame_interval = self.frame_interval
        else:
            frame_interval = max(round(fps / self.frame_rate), 1)

        fps = fps / frame_interval

        # Randomly select a start index for frame sequence
        start_idx = 0 if self.fixed_start_frame else random.randint(0, max(0, total_frames - self.sample_frames * frame_interval))

        def resize(frame):
            if self.width is not None and self.height is not None:
                return cv2.resize(frame, (self.width, self.height))
            if self.calibration_img_size is not None: # return the longest side to be calibration_img_size
                h, w = frame.shape[:2]
                if h > w:
                    new_h = self.calibration_img_size
                    new_w = int(w * self.calibration_img_size / h)
                else:
                    new_w = self.calibration_img_size
                    new_h = int(h * self.calibration_img_size / w)
                return cv2.resize(frame, (new_w, new_h))
            else:
                return frame

        for _ in range(start_idx):
            cap.grab()
        while True:
            flag, frame = cap.read()
            if not flag:
                break
            frame = torch.tensor(resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            frames.append(frame)

            if self.dense_calibration:
                frames_dense.append(frame)
            if len(frames) == self.sample_frames:
                break

            for j in range(frame_interval - 1):
                if self.dense_calibration:
                    flag, frame = cap.read()
                    if not flag:
                        frames_dense = frames_dense[:-j] # remove the appended frames if the video ends
                        break
                    frame = torch.tensor(resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    frames_dense.append(frame)
                else:
                    cap.grab() # fast forward to the next frame

        cap.release()

        if len(frames) < self.sample_frames:
            # append the last frame until we have enough frames
            for _ in range(self.sample_frames - len(frames)):
                frames.append(frames[-1])
                if self.dense_calibration:
                    frames_dense.extend([frames_dense[-1]] * frame_interval)

        video = torch.stack(frames).permute(0, 3, 1, 2).float() / 127.5 - 1 # Normalize to [-1, 1]

        if self.rotation: # roll along the width axis by a random amount, as videos are in 360 format
            rotate_pixels = random.randint(0, self.width)
            video = torch.roll(video, rotate_pixels, dims=-1) # (T, C, H, W)

        ret['video'] = video
        ret['path'] = video_path
        ret['frame_rate'] = fps
        # hard-coded for now

        if self.dense_calibration:
            ret['frames_dense'] = torch.stack(frames_dense).permute(0, 3, 1, 2).float() / 127.5 - 1

        if self.cached_motion:
            # category, video_id, clip = video_path.split('/')[-3:]
            video_name = os.path.basename(video_path)
            cached_motion_path = os.path.join(self.cached_motion_path, 'motion', f'{video_name[:-4]}.txt')
            motion = np.loadtxt(cached_motion_path)
            cached_fov_path = os.path.join(self.cached_motion_path, 'fov', f'{video_name[:-4]}.txt')
            fov = np.loadtxt(cached_fov_path)
            ret['motion'] = motion # np array of shape (3, 25)
            ret['fov'] = fov # np array of shape (2,)

        with open(caption_path, 'r') as f:
            caption = f.read().strip()
        ret['caption'] = caption

        return ret
        
import cv2
import numpy as np
import torch

def detect_vertical_lines(cap: cv2.VideoCapture, num_frames=10, device='cuda'):
    '''
    Detect vertical lines in the image.
    If any vertical line is detected, filter the video.
    '''

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # sample frames at random intervals
    positions = np.random.randint(0, num_frames, num_frames)

    frames = []
    for position in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = cap.read()
        frames.append(frame)
    cap.release()

    height, width = frames[0].shape[:2]

    frames = torch.from_numpy(np.array(frames)).to(device).unsqueeze(1).float() / 255. # (frame_count, 1, height, width)

    # Roll the frame 180 degrees (on the width dimension) to make the middle line the center
    frames = torch.roll(frames, frames.shape[-1] // 2, dims=-1)

    # keep the middle line
    frames = frames[:, :, :, width // 2 - 3: width // 2 + 3] # (frame_count, 1, H, 6)

    # apply sobel filter on width dimension
    sobel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).to(device)
    sobel_magnitude = torch.nn.functional.conv2d(frames.float(), sobel.unsqueeze(0).unsqueeze(0).float()).abs().squeeze(1) # (frame_count, H, 4)

    sobel_magnitude = sobel_magnitude.mean(dim=1) # (frame_count, 4)

    for sobel_magnitude_i in sobel_magnitude:
        if (sobel_magnitude_i[0] * 5 < sobel_magnitude_i[1] and sobel_magnitude_i[1] > 0.05) or \
            (sobel_magnitude_i[3] * 5 < sobel_magnitude_i[2] and sobel_magnitude_i[2] > 0.05):
            return False
            
    return True

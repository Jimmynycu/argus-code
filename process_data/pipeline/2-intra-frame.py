import cv2
import numpy as np
import torch
from equilib import equi2pers
import lpips
import math

def compute_lpips(cap: cv2.VideoCapture, num_frames=10, device='cuda'):
    '''
    Compute the LPIPS between the up and down halves of the frame.
    '''
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # sample frames at random intervals
    positions = np.random.randint(0, num_frames, num_frames)

    frames = []
    for position in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    frames = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1 # (frame_count, 3, height, width)
    height, width = frames.shape[2], frames.shape[3]

    lpips_vgg = lpips.LPIPS(net='vgg').to(device)

    # 1. calculate intra-frame difference
    left_right_diff = torch.abs(frames[:, :, :, :width // 2] - frames[:, :, :, width // 2:])
    up_down_diff = torch.abs(frames[:, :, :height // 2, :] - frames[:, :, height // 2:, :])

    left_right_lpips = lpips_vgg(frames[:, :, :, :width // 2], frames[:, :, :, width // 2:]).mean()
    up_down_lpips = lpips_vgg(frames[:, :, :height // 2, :], frames[:, :, height // 2:, :]).mean()

    # 2. calculate inter-frame difference 
    rolls = [0, 0, 0, 0, 0, 0]
    yaws = [0, 0, 0, math.pi / 2, math.pi, math.pi * 3 / 2]
    pitches = [0, math.pi / 2, -math.pi / 2, 0, 0, 0]
    fov_x = 90

    mask_differences = []
    mask_lpipses = []

    for idx, (roll, yaw, pitch) in enumerate(zip(rolls, yaws, pitches)):
        mask_difference, mask_lpips = 0, 0

        pers_frames = equi2pers(frames, fov_x=fov_x, rots=[{'roll': roll, 'yaw': yaw, 'pitch': pitch}] * len(frames), z_down=True)

        for i in range((num_frames - 1) // 2):
            mask_difference += torch.abs(pers_frames-torch.roll(pers_frames, i+1, dims=0)).mean()
            mask_lpips += lpips_vgg(pers_frames, torch.roll(pers_frames, i+1, dims=0)).mean()
        mask_differences.append(mask_difference.item() / ((num_frames - 1) // 2))
        mask_lpipses.append(mask_lpips.item() / ((num_frames - 1) // 2))

    mask_differences_avg = sum(mask_differences) / len(mask_differences)
    mask_lpipses_avg = sum(mask_lpipses) / len(mask_lpipses)

    if mask_differences_avg < 0.005 or mask_lpipses_avg < 0.002:
        return True
    elif left_right_diff.mean().item() < 0.05 or up_down_diff.mean().item() < 0.3:
        return True
    elif left_right_lpips.item() < 0.05 or up_down_lpips.item() < 0.3:
        return True
    else:
        return False


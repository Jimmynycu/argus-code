import cv2
import numpy as np

def filter_color_variance(cap: cv2.VideoCapture):
    '''
    Compute the variance of the frame at the center of the image.
    If any of the sampled frames has a variance less than 500, filter the video.
    '''
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = [0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]

    frames = []

    for position in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    # calculate the variance of the frame
    vars_all = [np.var(frame) for frame in frames]

    if min(vars_all) > 500:
        return True
    else:
        return False

def filter_black_pixels(cap: cv2.VideoCapture):
    '''
    Compute the number of black pixels in the frame.
    If all of the sampled frames has more than 20% black pixels, filter the video.
    '''
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = [0, num_frames // 3, 2 * num_frames // 3, num_frames - 1]
    frames = []

    for i in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    
    cap.release()

    # calcualte the number of black pixels
    stats = []
    for frame in frames:
        black_pixel_count = np.sum(frame <= 5)
        black_ratio = black_pixel_count / frame.size

        avg_intensity = np.mean(frame)
        stats.append((black_ratio, avg_intensity))

    if min(stats)[0] > 0.2:
        return True
    else:
        return False


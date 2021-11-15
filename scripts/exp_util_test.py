import numpy as np

# jzDXtd0__DM,180.54,5410

vid_dur = 180.54
vid_frame = 5410
sampling_sec = 0.5
frame_to_second= float(vid_dur)* int(float(vid_frame)*1./int(float(vid_dur))*sampling_sec)* 1./float(vid_frame)
print(frame_to_second)

# total frames sampled during 0.5 seconds ( based on actual total frames and actual total duration )
import numpy as np
import torch
# jzDXtd0__DM,180.54,5410

vid_dur = 180.54
vid_frame = 5410
sampling_sec = 0.5
frame_to_second= float(vid_dur)* int(float(vid_frame)*1./int(float(vid_dur))*sampling_sec)* 1./float(vid_frame)
print(frame_to_second)

# total frames sampled during 0.5 seconds ( based on actual total frames and actual total duration )
vgg_feats = torch.from_numpy(np.load('/home/gpuadmin/sandi/work_space/Detection-of-Traffic-Anomaly/dataset/features_self/0qfbmt4G8Rw_000306' + '.npy')).float()
slide_window_size = 480
total_frame = 128
img_feat = torch.FloatTensor(np.zeros((480, 4096)))

print("Vgg {} , img {}".format(vgg_feats.shape, img_feat[:128].shape))
torch.cat((vgg_feats), dim=1, out=img_feat[:min(total_frame, slide_window_size)])

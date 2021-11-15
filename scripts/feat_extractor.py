from tqdm import tqdm
import numpy as np
import torch 
from torch import optim, nn
from torchvision import models, transforms
import os
import skimage
import cv2
from pathlib import Path

def preprocess_frame(image, target_height=224, target_width=224):

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))


class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
		# Extract VGG-16 Feature Layers
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
		# Extract VGG-16 Average Pooling Layer
    self.pooling = model.avgpool
		# Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()
		# Extract the first part of fully-connected layer from VGG16
    self.fc = model.classifier[0]
  
  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 

# Initialize the model
model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(448),
  transforms.ToTensor()                              
])

# Will contain the feature

# Iterate each image
already_finish = os.listdir('../../Detection-of-Traffic-Anomaly/dataset/features_self/')
finish_list = []
for i in already_finish: 
  finish_list.append(Path(i).stem)
# print("list> ", finish_list)
base_frame_path = '../../Detection-of-Traffic-Anomaly/dataset/frames/'
for clip in os.listdir(base_frame_path):
  if clip not in finish_list:
    features = []
    frames_list = os.listdir(base_frame_path+clip)
    for frame in frames_list:
      frame = os.path.join(base_frame_path,clip,frame)
      img = cv2.imread(frame)
      img = transform(img)
      img = img.reshape(1, 3, 448, 448)
      img = img.to(device)
      with torch.no_grad():
        feature = new_model(img)
      features.append(feature.cpu().detach().numpy().reshape(-1))
    features = np.array(features)
    save_full_path = os.path.join('../../Detection-of-Traffic-Anomaly/dataset/features_self', clip + '.npy')
    np.save(save_full_path, features)
  elif clip in finish_list:
    print("Already finished > ", clip)
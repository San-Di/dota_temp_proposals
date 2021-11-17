import numpy as np
import torch
from torch.utils.data import DataLoader


train_loader = DataLoader([10,20,30,40,50],
                              batch_size=2,
                              shuffle=True,
                              num_workers=3)

a = next(iter(train_loader))
print(a)
a = next(iter(train_loader))
print(a)
a = next(iter(train_loader))
print(a)
a = next(iter(train_loader))
print(a)
a = next(iter(train_loader))
print(a)
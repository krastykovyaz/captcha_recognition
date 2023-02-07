import os
import numpy as np
import torch
import os
import torch.nn.functional as F
import torch.optim as optim


# from torch.utils.data import Dataset, Dataloader
from torchvision.models import resnet18



BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
CLIP_NORM = 5
DATA_PATH = 'samples/'
DEVICE = torch.device('cude' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 21
print(f'Device: {DEVICE}')

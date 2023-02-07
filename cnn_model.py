"""module for cnn model class"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class CRNN(nn.Module):
    def __init__(self, num_chars, rnn_hidden_size=256, dropout=0.1):
        super(CRNN, self).__init__()
        self.num_chars = num_chars
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout = dropout
        self.resnet = resnet18(weight=None)







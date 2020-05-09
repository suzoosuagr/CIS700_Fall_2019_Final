import os
import numpy as np
import random 
import datetime
import torch

from tensorboardX import SummaryWriter


# fix random seed.
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)

# device
if not torch.cuda.is_available():
    device = 'cpu'
else:
    device = 'cuda'
device = torch.device(device)

def train_epoch():
    pass


def test_epoch():
    pass









import torch
import torch.nn.functional as F

softmax_helper = lambda x: F.softmax(x, 1)

sigmoid_helper = lambda x: F.sigmoid(x)

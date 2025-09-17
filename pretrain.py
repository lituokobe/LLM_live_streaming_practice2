import os
import sys
import argparse
import time
import math
import torch
import warnings
import torch.distributed as dist # for distribution calculation
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model import MyModelConfig, MyModelForCausalLM
from dataset import PretrainDataset
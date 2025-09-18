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

warnings.filterwarnings("ignore")

def Logger(content):
    # if not distributed training, print on the single machine
    # if distributed training, print on the master machine
    if not ddp or dist.get_rank() ==0:
    print(content)

def get_lr(current_step, total_steps, lr):
    """
    function to adjust learning rate in the training. We use cos scheduler.
    Args:
        - current_step: current iteration
        - total_steps: total iteration neede
        - lr: initial learning rate
    """
    return lr/10 + 0.5*lr*(1+math.cos(math.pi*current_step/total_steps))

def init_model(lm_config):
    # read a readily available tokenizer from huggingface
    tokenizer = AutoTokenizer.from_pretrained('./model')
    # use our own class to initialize a model
    model = MyModelForCausalLM(lm_config).to(args.device)
    Logger(f"Trainable parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} millions.")
    return model, tokenizer

def init_distributed_model(args):
    if not ddp: return

    global ddp_local_rank, DEVICE # maek them a global variable

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    # make the code run on the local device, meaning a specific GPU (local rank) on a specific machine (global rank)
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)
def train_epoch(epoch):



if __name__ == "__main__":
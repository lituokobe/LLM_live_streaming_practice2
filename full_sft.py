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
from dataset import SFTDataset

warnings.filterwarnings("ignore")

"""
Full SFT will retrain all the parameters of the model, including the experts in MoE layers.
So the code is very similar to pretrain.py.

We adapt the model trained on general text to a Q&A chatbot which is fine-tuned with Q&A data.
"""

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
        - total_steps: total iteration needed
        - lr: initial learning rate
    """
    return lr/10 + 0.5*lr*(1+math.cos(math.pi*current_step/total_steps))

# thef function of init_model is the only part changed from pretrain.py
def init_model(lm_config):
    # read a readily available tokenizer from huggingface
    tokenizer = AutoTokenizer.from_pretrained('./model')
    # use our own class to initialize a model
    model = MyModelForCausalLM(lm_config) # at this time, the parameters are randomly initialized
    
    # load the pretrained model parameters
    moe_path = "_moe" if lm_config.use_moe else ""
    ckp = f"./pretrained_model/pretrain_{lm_config.hidden_size}{moe_path}.pth"
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False) # load the pretrained model parameters, strict=False means we can load part of the parameters
    Logger(f"Trainable parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} millions.")
    model = model.to(args.device)
    
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
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    # reduction="none" means we don't want to sum the loss, we want the loss of each sample, because we need to apply loss mask
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch*iter_per_epoch + step, args.epochs*iter_per_epoch, args.learning_rate)

        for param_group in optimizer.param_groups:
            # set learning rate for each layer that is optimized
            param_group["lr"] = lr
        # ctx, either based on CPU or GPU, for mixed precision
        with ctx:
            res = model(X) # forward pass for prediction results
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum() # only keep the loss of the samples that are not padding, get average loss per token
            loss += res.aux_loss # Add moe's auxiliary loss
            loss = loss/args.accumulation_steps # accumulation of gradient, this is an optimization technique
            """
            loss = loss / args.accumulation_steps ensures that when you're accumulating gradients over multiple mini-batches,
            the overall gradient magnitude remains consistent with what you'd get from a single large batch.
            It’s a critical adjustment for stable training.
            """
        # user mixed precision (FP32, FP16) training for backward propagation
        # It's easy to have gradient disappearance when using FP16 training
        scaler.scale(loss).backward() #scale up the loss

        if (step+1)%args.accumulation_steps == 0:
            """
            accumulation of gradient means after multiple forward propagation to have loss and back propagation to have gradient,
            then we use the gradient to update the parameters
            """
            # gradient is on optimizer, we unscale the gradient because we scaled up the loss
            scaler.unscale_(optimizer)
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # apply the gradient to update parameters
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step%args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                "Epoch: [{}/{}]({}/{}), loss:{:.3f}, learning rate: {:.12f}, epoch time {} min:".format(
                    epoch+1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item()*args.accumulation_steps,
                    optimizer.param_groups[-1]["lr"],
                    spend_time/(step+1)*iter_per_epoch//60 - spend_time//60
                )
            )
        if (step+1)%args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            # only save the model in the only machine or the main machine
            model.eval()
            moe_path = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth"
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # if parallel training, we need to get the state dict of the model
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            # save it in half precision, FP16
            state_dict = {k:v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyModel Pretraining")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--epochs", type=int, default=1) # for better training, use 2-6 epoches
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action = "store_true") # if this parameter shows up, it will be true, otherwise false
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    # parser.add_argument("--use_moe", type=bool, default=False)
    parser.add_argument("--use_moe", type=bool, default=True)
    parser.add_argument("--data_path", type=str, default="./data/sft_mini_512.jsonl")
    args = parser.parse_args()
    lm_config = MyModelConfig(
        hidden_size = args.hidden_size,
        num_hidden_layers = args.num_hidden_layers,
        use_moe = args.use_moe
    )
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True) #exist_ok=True means even if the directory exists, it will not raise an error
    os.makedirs(args.out_dir, exist_ok=True)

    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # mixed precision training
    # if device_type == "cpu" ctx equals nothing
    # torch.cuda.amp.autocast() is a context manager that tells PyTorch to automatically choose
    # the most efficient floating-point precision (FP16 or FP32) for different operations during the forward pass.
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1))!=-1
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # set random seed to reproduce the same results
    base_seed = 128
    torch.manual_seed(base_seed) #for CPU
    torch.cuda.manual_seed(base_seed) #for GPU

    # Initialization for distributed calculation
    if ddp:
        init_distributed_model()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    # Initialize the model
    model, tokenizer = init_model(lm_config)
    # We also changed the dataset class to fit the SFT data structure
    # In this example, the SFT data is a collection of jsons that inclue Q and A
    # The original pretraining data is a only collection of texts
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len) 
    train_sampler = DistributedSampler(train_ds) if ddp else None
    # Read the sample one by one, return the data batch by batch
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
        sampler=train_sampler,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16","bfloat16"]))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)  

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)

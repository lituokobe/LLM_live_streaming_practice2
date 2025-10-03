import os
import sys
import argparse
import time
import math
import torch
import warnings
import torch.distributed as dist # for distribution calculation
from torch import optim, nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model import MyModelConfig, MyModelForCausalLM
from dataset import DPODataset

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
        - total_steps: total iteration needed
        - lr: initial learning rate
    """
    return lr/10 + 0.5*lr*(1+math.cos(math.pi*current_step/total_steps))

def logits_to_probs(logits, labels):
    # Convert the log probability generated from the model to probability distribution
    # logits shape : (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2) # softmax first, then log
    # get the probabilities from the correct label
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs

# def dpo_loss(ref_probs, probs, mask, beta):
#     # ref_probs: probabilities from reference model's deduction
#     # probs: probabilities from the model that needs DPO training
#     # mask: question+answer that is relevant to loss calculation - aka answer
#     # beta: hyperparameter in DPO loss function
#     # ref_probs and probs have the same shape (batch_size, seq_len)
#     seq_lengths = mask.sum(dim=1, keepdim=True) # (batch_size, 1)
#     ref_probs = (ref_probs*mask).sum(dim = 1)/seq_lengths.squeeze()
#     probs = (probs*mask).sum(dim = 1)/seq_lengths.squeeze()

#     # split chosen and rejected
#     batch_size = ref_probs.shape[0]
#     chosen_ref_probs = ref_probs[:batch_size // 2]
#     rejected_ref_probs = ref_probs[batch_size // 2:]
#     chosen_probs = probs[:batch_size // 2]
#     rejected_probs = probs[batch_size // 2:]

#     pi_logratios = chosen_probs - rejected_probs
#     ref_logratios = chosen_ref_probs - rejected_ref_probs
#     logits = pi_logratios - ref_logratios

#     loss = -F.logsigmoid(beta*logits)
#     return loss.mean()
def dpo_loss(ref_probs, probs, mask, beta):
    
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
    
    # Average the regular probabilities, not the log probabilities
    ref_probs_avg = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    probs_avg = (probs * mask).sum(dim=1) / seq_lengths.squeeze()
    
    # Convert back to log space for the ratio calculation
    ref_probs_log = torch.log(ref_probs_avg.clamp(min=1e-12))
    probs_log = torch.log(probs_avg.clamp(min=1e-12))
    
    # split chosen and rejected
    batch_size = ref_probs_log.shape[0]
    chosen_ref_probs = ref_probs_log[:batch_size // 2]
    rejected_ref_probs = ref_probs_log[batch_size // 2:]
    chosen_probs = probs_log[:batch_size // 2]
    rejected_probs = probs_log[batch_size // 2:]

    pi_logratios = chosen_probs - rejected_probs
    ref_logratios = chosen_ref_probs - rejected_ref_probs
    logits = pi_logratios - ref_logratios

    loss = -F.logsigmoid(beta * logits)
    return loss.mean()

# the function of init_model is the part different from pretrain.py
def init_model(lm_config):
    # read a readily available tokenizer from huggingface
    tokenizer = AutoTokenizer.from_pretrained('./model')
    # use our own class to initialize a model
    model = MyModelForCausalLM(lm_config) # at this time, the parameters are randomly initialized
    
    # load the pretrained model parameters
    moe_path = "_moe" if lm_config.use_moe else ""
    ckp = f"./SFT_model/full_sft_{lm_config.hidden_size}{moe_path}.pth"
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False) # load the pretrained model parameters, strict=False means we can load part of the parameters
    Logger(f"Trainable parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} millions.")
    model = model.to(args.device)

    # initialize the reference model
    ref_model = MyModelForCausalLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)
    ref_model.to(args.device)

    return model, ref_model, tokenizer

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
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        x_chosen = batch["x_chosen"].to(args.device)
        y_chosen = batch["y_chosen"].to(args.device)
        mask_chosen = batch["mask_chosen"].to(args.device)
        x_rejected = batch["x_rejected"].to(args.device)
        y_rejected = batch["y_rejected"].to(args.device)
        mask_rejected = batch["mask_rejected"].to(args.device)
        x = torch.cat([x_chosen, x_rejected], dim=0) # stack the sample to have only one x
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch*iter_per_epoch + step, args.epochs*iter_per_epoch, args.learning_rate)

        for param_group in optimizer.param_groups:
            # set learning rate for each layer that is optimized
            param_group["lr"] = lr
        # ctx, either based on CPU or GPU, for mixed precision
        with ctx:
            with torch.no_grad(): # ref_model won't be trained
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask

            # forward propagation
            outputs = model(x)
            logits = outputs.logits

            probs = logits_to_probs(logits, y)
            probs = probs * mask

            loss = dpo_loss(ref_probs, probs, mask, beta=0.1)

            # if torch.isnan(loss).any() or torch.isinf(loss).any():
            #     Logger(f"NaN/Inf in loss: {loss}")
            #     Logger(f"ref_probs range: {ref_probs.min():.6f} to {ref_probs.max():.6f}")
            #     Logger(f"probs range: {probs.min():.6f} to {probs.max():.6f}")
            #     continue

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
            ckp = f"{args.save_dir}/dpo_{lm_config.hidden_size}{moe_path}.pth"
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
    parser = argparse.ArgumentParser(description="MyModel DPO")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--epochs", type=int, default=1) # for better training, use 2-6 epoches
    parser.add_argument("--batch_size", type=int, default=8)
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
    parser.add_argument("--use_moe", type=bool, default=True)
    parser.add_argument("--data_path", type=str, default="./data/dpo.jsonl")
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
    model, ref_model, tokenizer = init_model(lm_config)
    # We also changed the dataset class to fit the SFT data structure
    # In this example, the SFT data is a collection of jsons that inclue Q and A
    # The original pretraining data is a only collection of texts
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len) 
    train_sampler = DistributedSampler(train_ds) if ddp else None
    # Read the sample one by one, return the data batch by batch
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
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

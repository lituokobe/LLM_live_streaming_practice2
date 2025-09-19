# import libraries
import math
from typing import Optional, List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

# Define MyModelConfig class
class MyModelConfig(PretrainedConfig):
    model_type = "MyModel"
    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = "silu", # activation function for activation layer
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32678,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-5,
            rope_theta: float = 10000.0,
            flash_attn: bool = False,
            #MEO parameters
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = "softmax",
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
     )
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        #MEO parameters
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob

# Define RMSNorm class
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x *torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

# Pre-calculate cos and sin for each position
def precompute_freqs_cis(dim: int, end: int = int(32*1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # [: (dim // 2)] means dim must be even
    t = torch.arange(end, device=freqs.device)
    #m*theta
    freqs = torch.outer(t, freqs).float()
    #cos(m*theta)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    #sin(m*theta)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin
# apply positional embedding
def apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin, position_ids=None, unsqueeze_dim = 1):
    def rotate_half(x):
        # arr[..., 2:4] = arr[:, :, :, 2:4]
        # categorize in the input to groups of 2, but not (x1, x2) (x3, x40... but (x1, x_d/2) (x2, x_d/2+1)....
        # this is RoPE done by HuggingFace and Qwen
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_emb = q * freqs_cos.unsqueeze(unsqueeze_dim) + (rotate_half(q) * freqs_sin.unsqueeze(unsqueeze_dim))
    k_emb = k * freqs_cos.unsqueeze(unsqueeze_dim) + (rotate_half(k) * freqs_sin.unsqueeze(unsqueeze_dim))
    return q_emb, k_emb

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            x[:, :, :, None, :].expand(bs, slen, num_key_value, n_rep, head_dim)
            .reshape(bs, slen, num_key_value * n_rep, head_dim)
        )

# Self Attention
class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

    def forward(self,
                x: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                past_key_value = None,
                use_cache: bool = False,
                attention_mask: torch.Tensor = None):
        bsz, seq_len, _ = x.size()
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        #RoPE
        pre_cos, pre_sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, pre_cos[:seq_len], pre_sin[:seq_len])

        #KV cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        if use_cache:
            past_kv = (xk, xv)
        else:
            past_kv = None

        xq, xk, xv = (
            xq.transpose(1,2),
            repeat_kv(xk, self.n_rep).transpose(1,2),
            repeat_kv(xv, self.n_rep).transpose(1,2)
        )

        # Self-attention
        scaled_scores = (xq@xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores + mask
        masked_scores = scaled_scores + torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=scores.device)
        ) # masked (make it to infinite negative) to all values above the diagonal
        masked_scores = masked_scores.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 0 attention mask will become -1e9
            # 1 attention mask will become 0
            # Anything below 1 or near one will become 0 or below, which will be ignored in the softmax
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            masked_scores = masked_scores + extended_attention_mask

        scores = F.softmax(masked_scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = scores @ xv
        output = output.transpose(1,2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.intermidiate_size is None:
            intermediate_size = int(config.hidden_zie*8/3)
            config.intermediate_size = 64*((intermediate_size+64-1)//64)

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act] #Non linear transformation, like Swish
    def forward(self, x):
        return self.dropout(self.down_proj(self.up_proj(x)*self.act_fn(self.gate_proj(x))))

# MOE gate
class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts # total amounts of experts
        self.scoring_func = config.scoring_func # scoring function, usually softmax

        # to make MoE more balanced (avoid always favoring several experts with long tailed selection)
        self.alpha = config.aux_loss_alpha # weight of the auxiliary loss
        self.seq_aux = config.seq_aux # calculate loss of the balance level of MoE (token leval and sequence level)

        self.norm_topk_prob = config.norm_topk_prob # if normalize the probabilities of top K
        self.gating_dim = config.hidden_size # dimension of input vectors

        # Define a learnable gate matric, shape: [n_routed_experts, gating_dim]
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # Initialize the gate matrix
        self.reset_parameters()

    def register_parameter(self):
        import torch.nn.init as init
        init.kaiming_uniform(self.weight, a = math.sqrt(5))

    def forward(self, hidden_states):
        # This is the core logic, input is a hidden state, output is result of expert allocation and auxiliary loss
        bsz, seq_len, h = hidden_states.size()
        # make it 2-dimensional, for the ease to process each token [bsz*seq_len, h]
        hidden_states = hidden_states.view(-1, h)
        # calculate original logits of tokens for each expert, shape[total_tokens, n_routed_experts]
        logits = F.linear(hidden_states, self.weight, bias = None) # F.linear(x, w) == torch.matmul(x, w.T)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim = -1)
        else:
            raise NotImplementedError('Invalid scoring function')

        # for each token, select the expert with highest score
        topk_weight, topk_idx = torch.topk(scores, k = self.top_k, dim = -1, sorted =False)

        # whether use norm_topk_prob. Is so, normalize the topk_prob
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim = -1, keepdim = True) + 1e-20
            topk_weight = topk_weight / denominator

        # if in training mode and auxiliary loss is activated, we need to calculate the auxiliary loss
        if self.training and self.alpha > 0:
            scores_for_aux = scores # scores of all experts before topk selection
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idex.view(bsz, -1) # Flattened topk expert index
            if self.seq_aux:
                # calculate auxiliary loss for sequence level
                # view a sequence as a whole, is the whole sequence only use one expert, then punish the sequence to encourage to use multiple experts
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # built a frequency matrix of experts being chosen
                ce = torch.zeros((bsz, self.n_routed_experts), device = hidden_states.device)
                # count the frequency of each expert
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len*aux_topk)).div_(seq_len*aux_topk/self.n_routed_experts)
                # multiply the average as the loss
                aux_loss = (ce*scores_for_seq_aux.mean(dim = 1)).sum().mean()*self.alpha
            else:
                # calculate auxiliary loss for token level
                # To see what expert is chosen by each token, punish the token to encourage it to choose multiple experts
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes = self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                # calculate expert's average score
                Pi = scores_for_aux.mean(0)
                # calculate the frequency of each expert being chosen
                fi = ce*self.n_routed_experts
                # calculate the multiplied result as loss
                aux_loss = (Pi*fi).sum()*self.alpha

        return topk_idx, topk_weight, aux_loss

# MoE feedforward class
class MoEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0: # shared experts that is selected everytime exists. This is Deepseek mechanism.
            self.shared_experts = nn.ModuleList([
                FeedForward(config) for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # Use gate mechanism to select expert
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # duplicate each token to the amount of num_experts_per_tok
            # this is to pass each token to selected top-k experts at the same time
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim = 0)
            # create a tensor that has the same shape of x, but with empty values in the type of float16, to store the result of each experts
            y = torch.empty_like(x, dtype = torch.float16)
            for i, expert in enumerate(self.experts):
                # flat_topk_idx is an index tensor, meaning which expert is allocated to each token
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            # reshape the output
            # use token_weight to have a weighted output of all the experts' results
            y = (y.view(*topk_weight.shape, -1)*topk_weight.unsqueeze(-1).sum(dim = 1))
            # restore the output shape
            y = y.view(*orig_shape)
        else:
            # In inference mode, we only use the more effective function "moe_infer" as the MOE part
            # This is to reduce redundancy in ram or calculation. e.g. it will combine several tokens and process together
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # if we used shared experts, they will be applied to all tokens
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        # the loss will be added to total_loss. total+loss = task_loss + config.aux_loss_coeff*model.aux
        self.aux_loss = aux_loss

        return y

    @torch.no_grad() # same as "with torch.no_grad...."
    def moe_infer(self, x, flat_expert_indices, flat_expert_weight):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        """
        Example: 
        tokens_per_expert = [6, 15, 20, 26] # the 4 values represents token amount processed by 4 experts
        """
        tokens_idxs = idxs//self.config.num_experts_per_tok
        """
        Example:
        token_idx = [3, 7 ,19, 21, 24, 25, 4, 5, 6, 10, 11, 12...] 
        # first 6 tokens_idx[:6] belong to expert 0
        # each token may be processed by multiple experts, this is decided by config.num_experts_per_tok
        """
        for i, end_idx in enumerate(tokens_per_expert):
            # calculate token's start index processed by current expert
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            # if no token is alloated to this expert, then we skip this expert
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = tokens_idxs[start_idx:end_idx]
            # get the token's embedding from the original input of x
            expert_tokens = x[exp_token_idx]
            # input in current expert and forward propagate
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # add weights to expert's output
            expert_out.mul_(flat_expert_weight[idxs[start_idx:end_idx]])
            # add expert output to the final output
            expert_cache.scatter_add_(0, exp_token_idx.view(-1,1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

# define blocks
class MyModelBlock(nn.Module):
    def __init__(self, layer_id: int, config: MyModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)
        self.layer_id = layer_id
        self.input.layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MoEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_values = None,
                use_cache = False, attention_mask = None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(self.input.layernorm(hidden_states), position_embeddings,
                                                          past_key_values, use_cache, attention_mask)

        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MyModel(nn.Module):
    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embd_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # make layers of blocks
        self.layers = nn.ModuleList([MyModelBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim = config.hidden_size//config.num_attention_heads,
                                                    end = config.max_position_embeddings,
                                                    theta = config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent = False) # register buffer to avoid computing again
        self.register_buffer("freqs_sin", freqs_sin, persistent = False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                user_cache: bool = False,
                **kwargs):
        batch_size, seq_len = input_ids.shape
        past_key_values = past_key_values or [None]*len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embd_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos: start_pos + seq_len]
        )

        presents = []

        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # layer is MyModelBlock
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache = user_cache,
                attention_mask = attention_mask
            )
            # store key_value from attention calculation, then put it in cache
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            # layer.mlp takes the mlp out from block, it could be MoEFeedForward or FeedForward
            layer.mpl.aux_loss for layer in self.layers
            if isinstance(layer.mlp, MoEFeedForward)
        )

        return hidden_states, presents, aux_loss

class MyModelForCausalLM(PreTrainedModel, GenerationMixin):
    # We use huggingface class so that it's easy to use hugging face library to load our model
    config_class = MyModelConfig

    def __init__(self, config: MyModelConfig = None):
        self.config = config or MyModelConfig()
        super().__init__(self.config)
        self.model = MyModel(self.config)
        # This is output layer
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False)
        # use shared weight for tensors with same (transposed) dimension, this reduce the parameter amount
        self.model.embd_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                user_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = None,
                **kwargs):
        # h is the output of stacked blocks
        h, past_kvs, aux_loss = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            user_cache = user_cache,
            **args
        )

        # if logits_to_keep is an int, we slice the last digits of the number of the int; if it's a tensor, we keep the slice of the numbers in this tensor
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        logits = self.lm_head(h[:, slice_indices, :])

        # Huggingface style to set output
        self.OUT.__setitem__("last_hidden_state", h)
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("past_key_values", past_kvs)
        self.OUT.__setitem__("aux_loss", aux_loss)
        return self.OUT
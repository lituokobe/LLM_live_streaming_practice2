import torch
from torch import optim, nn

# Define lora network
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # initialize A and B
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()
    def forward(self, x):
        return self.B(self.A(x))

def apply_lora(model, rank=8):
    # iterate over all modules in the model
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # replace the linear layer with a LoRA layer. xq, xk, xv are all linear layers
            # module.weight.shape[0] == module.weight.shape[1] is optional. it means a square matrix
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank).to(model.device)

            setattr(module, 'lora', lora)
            original_forward = module.forward

            # logic for forward propagation after lora
            def forward_with_lora(x, layer1 = original_forward, layer2 = lora):
                # combine the original result with the lora result. This is the classic LoRA method
                return layer1(x) + layer2(x)
            module.forward = forward_with_lora

def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # rename the lora parameter by removing the loara prefix
            lora_state = {k.replace(f"{name}.lora.", ""): v for k, v in state_dict.items() if f"{name}.lora" in k}
            module.lora.load_state_dict(lora_state)

def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # save the lora parameters and adding lora prefix
            lora_state = {f"{name}.lora.{k}": v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)

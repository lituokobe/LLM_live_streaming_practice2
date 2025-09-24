import os
import sys
import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from model import MyModelConfig, MyModelForCausalLM

warnings.filterwarnings("ignore", category=UserWarning)

def convert_torch2transformers_mymodel(torch_path, transformers_path, dtype=torch.bfloat16):
    """
    Convert a custom PyTorch model to Hugging Face Transformers format.
    It can be used for both dense and Mixture of Experts (MoE) models.
    Args:
        - torch_path: path to the source .pth file.
        - transformers_path: directory where the converted model will be saved.
        - dtype: data type to which the model should be converted (default: torch.bfloat16).
    Returns:
        None
    """
    # register_for_auto_class() allows the custom model to be recognized by Hugging Face's AutoModel classes.
    # This means you can:
    # - use AutoModelForCausalLM.from_pretrained(...) and it will return MyModelForCausalLM instance.
    # - save and load the model using Hugging Face’s save_pretrained() and from_pretrained() APIs.
    # - integrate with Hugging Face tools like Trainer, pipeline, and transformers-cli — assuming your model follows expected interfaces.
    MyModelConfig.register_for_auto_class()
    MyModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    lm_model = MyModelForCausalLM(lm_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(torch_path, map_location=device) # load the state_dict from the .pth file we pretrained
    lm_model.load_state_dict(state_dict, strict=False) # load the state_dict into the model
    lm_model = lm_model.to(dtype=dtype) # convert the model to the specified dtype
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad) # get the number of trainable parameters
    print(f"Trainable parameters in the model: {model_params/1e6:.2f} millions, or {model_params/1e9:.2f} billions.")

    lm_model.save_pretrained(transformers_path, safe_serialization = False) # save the model in transformers format
    # safe_serialization = True (default) -> save as safetensors, False -> save as pickle

    tokenizer = AutoTokenizer.from_pretrained("./model/") 
    tokenizer.save_pretrained(transformers_path) # save the tokenizer in transformers format
    print(f"MyModel and tokenizer saved to {transformers_path} in the format of Transformers-MyModel.")


def convert_torch2transformers_llama(torch_path, transformers_path, dtype=torch.bfloat16):
    # This for converting dense model like Llama (e.g. Qwen3) from .pth to transformers format
    # LlamaForCausalLM is compatical to third party ecosystem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(torch_path, map_location=device)
    # Create a LlamaConfig instance with the default huggingface llama class
    llama_config = LlmaConfig(
        vocab_size = lm_config.vocab_size,
        hidden_size = lm_config.hidden_size,
        intermediate_size = ((int(lm_config.intermediate_size*8/3) + 64 - 1)//64),
        num_hidden_layers = lm_config.num_hidden_layers,
        num_attention_heads = lm_config.num_attention_heads,
        num_key_value_heads = lm_config.num_key_value_heads,
        max_position_embeddings = lm_config.max_seq_len,
        rms_norm_eps = lm_config.rms_norm_eps,
        rope_theta = lm_config.rope_theta,
    )

    llam_model = LlamaForCausalLM(llama_config)
    llama.model.load_state_dict(state_dict, strict=False)
    llama_model = llama_model.to(dtype=dtype)
    llama_model.save_pretrained(transformers_path)

    tokenizer = AutoTokenizer.from_pretrained("./model/") 
    tokenizer.save_pretrained(transformers_path) # save the tokenizer in transformers format
    print(f"Llama model and tokenizer saved to {transformers_path} in the format of Transformers-Llama.")


if __name__ == "__main__":
    lm_config = MyModelConfig(hidden_size=512, num_hidden_layers=8, max_seq_len=512, use_moe=True)
    
    # the pretrained model was saved in ./out folder first
    # in order not to cover it by possible later training, I moved it ./pretrained_model folder
    torch_path = "./pretrained_model/pretrain_512_moe.pth"
    transformers_path = "./converted_model"

    convert_torch2transformers_mymodel(
        torch_path = torch_path,
        transformers_path = transformers_path, 
        dtype = torch.bfloat16
    )

    # convert_torch2transformers_llama(
    #     torch_path = torch_path,
    #     transformers_path = transformers_path, 
    #     dtype = torch.bfloat16
    # )

    
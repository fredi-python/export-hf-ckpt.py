import os

import torch
import transformers
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Model,
    GPTNeoModel,
)  # noqa: F402

BASE_MODEL = os.environ.get("BASE_MODEL", None)
lora_weights = os.environ.get("LORA_WEIGHTS", None)
assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

if isinstance(base_model, GPT2Model):
    transformer_layers = base_model.transformer
elif isinstance(base_model, GPTNeoModel):
    transformer_layers = base_model.model
else:
    raise ValueError(
        f"Unsupported base model class {type(base_model)}. "
        "Please use a GPT2Model or GPTNeoModel."
    )

first_weight = transformer_layers.h[0].attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    lora_weights,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.h[0].attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

AutoModelForCausalLM.save_pretrained(
    base_model, "./hf_ckpt", state_dict=deloreanized_sd, max_shard_size="400MB"
)

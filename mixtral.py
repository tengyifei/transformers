import torch
from transformers import AutoTokenizer, AutoConfig, MixtralForCausalLM

model_id = "mistralai/Mixtral-8x7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(
    model_id,
    vocab_size=len(tokenizer),
    torch_dtype=torch.bfloat16,
    num_hidden_layers=1,
    hidden_size=512,
    intermediate_size=2048,
    num_local_experts=4,
)
print(config)
model = MixtralForCausalLM(config)
print(f"Model parameters: {model.num_parameters()/2**20:.2f}M params")

output = model(torch.randint(512, (2, 128)))
print(output.logits.shape)
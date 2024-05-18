import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

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

# device = xm.xla_device()
device = 'cpu'
torch.manual_seed(42)

# This is a custom config to enable the static mode of expert computation.
config.static=False
model = MixtralForCausalLM(config).to(device)
print(f"Model parameters: {model.num_parameters()/2**20:.2f}M params")

output = model(torch.arange(8).view(1, 8).to(device))
# xm.mark_step()
print(output.logits.shape)
print(output.logits)
# print(met.metrics_report())

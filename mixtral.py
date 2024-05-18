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

device = 'cpu'

config.static=False
torch.manual_seed(42)
dynamic_model = MixtralForCausalLM(config).to(device)
print(f"Model parameters: {dynamic_model.num_parameters()/2**20:.2f}M params")

# This is a custom config to enable the static mode of expert computation.
config.static=True
torch.manual_seed(42)
static_model = MixtralForCausalLM(config).to(device)
print(f"Model parameters: {static_model.num_parameters()/2**20:.2f}M params")

input_sizes = [8, 128, 256, 512, 1024]
for input_size in input_sizes:
    input = torch.randint(128, ((2, input_size // 2))).to(device)
    static_output = static_model(input)
    print(static_output.logits.shape)
    dynamic_output = dynamic_model(input)
    print(dynamic_output.logits.shape)
    assert torch.allclose(static_output.logits, dynamic_output.logits, atol=1e-6), "logits are not equal"

device = xm.xla_device()
model = static_model.to(device)
output = model(torch.randint(128, ((2, 128))).to(device))
xm.mark_step()
print(met.metrics_report())

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import numpy as np

from transformers import AutoTokenizer, AutoConfig, MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

model_id = "mistralai/Mixtral-8x7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(
    model_id,
    vocab_size=len(tokenizer),
    torch_dtype=torch.bfloat16,
    num_hidden_layers=1,
    num_attention_heads=8,
    hidden_size=8,
    intermediate_size=16,
    num_local_experts=4,
)
print(config)
config.flash_attention = False

device = 'cpu'

config.static=False
config.gmm=False
config.gmm_stack=False
torch.manual_seed(42)
dynamic_model = MixtralForCausalLM(config).to(device)
# dynamic_model = MixtralSparseMoeBlock(config).to(device)
print(f"Model parameters: {dynamic_model.num_parameters()/2**20:.2f}M params")

active_weight = 0
for name, param in dynamic_model.named_parameters():
    if not param.requires_grad:
        continue
    numel = param.numel()
    print(f"{name}: {numel} params")
    if 'block_sparse_moe.experts' in name:
        active_weight += numel / config.num_local_experts * config.num_experts_per_tok
    else:
        active_weight += numel
print(f"Active weight: {active_weight/2**20:.2f}M params")

# This is a custom config to enable the static/gmm mode of expert computation.
# config.static=True
config.gmm_stack=True  # for cpu, it will use eager gmm.
torch.manual_seed(42)
static_model = MixtralForCausalLM(config).to(device)
# static_model = MixtralSparseMoeBlock(config).to(device)
print(f"Model parameters: {static_model.num_parameters()/2**20:.2f}M params")

tests = [
    {
        'm': 128,
        'k': 128,
        'n': 128,
        'num_groups': 1
    },
    {
        'm': 256,
        'k': 128,
        'n': 128,
        'num_groups': 1
    },
    {
        'm': 128,
        'k': 256,
        'n': 128,
        'num_groups': 8
    },
    {
        'm': 512,
        'k': 128,
        'n': 256,
        'num_groups': 2
    },
]

def group_sizes_strategy(m: int, num_groups: int) -> torch.Tensor:
    # Randomly sample the ends of the groups in the m-dimension. Let the fuzzer
    # sample with replacement so that it's possible to get zero-sized groups. Get
    # 'num_groups - 1' run ends. The final group will end at 'm'.
    ends_no_final = np.sort(
        np.array(
            [np.random.randint(low=0, high=m) for _ in range(num_groups - 1)],
            dtype=np.int32,
        ),)
    ends = np.concatenate([ends_no_final, np.array([m], dtype=np.int32)])

    # Calculate the run starts by shifting ends 1 to the right. The first run
    # starts at zero.
    starts = np.concatenate([np.zeros(1, dtype=np.int32), ends_no_final])
    return torch.from_numpy(ends - starts).to(torch.int32)


for test in tests:
    from transformers.models.mixtral.modeling_mixtral import Gmm

    num_groups = test['num_groups']
    k = test['k']
    m = test['m']
    n = test['n']
    lhs_dtype = rhs_dtype = torch.bfloat16
    print(f"Running test with m={m}, k={k}, n={n}, num_groups={num_groups}")

    lhs = torch.rand(m, k, dtype=lhs_dtype, requires_grad=True)
    rhs = torch.rand(num_groups, k, n, dtype=rhs_dtype, requires_grad=True)
    group_sizes = group_sizes_strategy(m=m, num_groups=num_groups)
    lhs.retain_grad()
    rhs.retain_grad()

    ref_out = Gmm._eager_gmm(lhs, rhs, group_sizes)
    ref_out.sum().backward()

    ref_out_backward = torch.ones_like(ref_out)
    grad_lhs, grad_rhs = Gmm._eager_gmm_backward(
        ref_out_backward, lhs, rhs,
        group_sizes)

    assert torch.allclose(lhs.grad, grad_lhs.cpu())
    assert torch.allclose(rhs.grad, grad_rhs.cpu())

for test in tests:
    from transformers.models.mixtral.modeling_mixtral import Gmm

    num_groups = test['num_groups']
    k = test['k']
    m = test['m']
    n = test['n']
    lhs_dtype = rhs_dtype = torch.bfloat16
    print(f"Running another test with m={m}, k={k}, n={n}, num_groups={num_groups}")


    # Create TopK
    top1 = torch.randint(0, num_groups, (m, 1)).to(device)
    top2 = torch.randint(0, num_groups, (m, 1)).to(device)
    top = torch.cat([top1, top2], dim=1)

    lhs = torch.rand(m, k, dtype=lhs_dtype, requires_grad=True)
    w1 = torch.rand(num_groups, k, n, dtype=rhs_dtype, requires_grad=True)
    w3 = torch.rand(num_groups, k, n, dtype=rhs_dtype, requires_grad=True)
    w2 = torch.rand(num_groups, n, k, dtype=rhs_dtype, requires_grad=True)
    lhs.retain_grad()
    w1.retain_grad()
    w2.retain_grad()
    w3.retain_grad()

    class Empty:
        pass
    context = Empty()
    context.save_for_backward = lambda *args: args
    ref_out = Gmm.forward(Gmm, lhs, top, w1, w2, w3)
    ref_out.sum().backward()
    grad_lhs, grad_w1, grad_w2, grad_w3 = lhs.grad, w1.grad, w2.grad, w3.grad

    lhs.grad = w1.grad = w2.grad = w3.grad = None
    out = Gmm.apply(lhs, top, w1, w2, w3)
    out.sum().backward()

    assert torch.allclose(lhs.grad, grad_lhs)
    assert lhs.grad is not grad_lhs
    assert torch.allclose(w1.grad, grad_w1)
    assert w1.grad is not grad_w1
    assert torch.allclose(w2.grad, grad_w2)
    assert w2.grad is not grad_w2
    assert torch.allclose(w3.grad, grad_w3)
    assert w3.grad is not grad_w3

# Grads are bastards to compare, set the precisou to 1e-4.
def compare_tensors(t1, t2):
    result = torch.allclose(t1, t2, atol=1e-4, rtol=1e-4)
    if result:
        return True
    else:
        print(t1.shape)
        # print(t1)
        # print(t2)
        diff = torch.abs(t1 - t2)
        print(f"Max diff: {diff.max()}")
        return False

# Grads are bastards to compare, use smaller batch sizes.
input_sizes = [8, 128, 256]
for input_size in input_sizes:
    input = torch.randint(128, ((2, input_size // 2))).to(device)
    # input = torch.randn((2, input_size // 2, 8), requires_grad=True).to(device)
    static_output = static_model(input).logits
    print(static_output.shape)
    # print(static_output.logits)
    dynamic_output = dynamic_model(input).logits
    print(dynamic_output.shape)
    # print(dynamic_output.logits)
    assert torch.allclose(static_output, dynamic_output, atol=1e-6), "logits are not equal"

    static_output.sum().backward()
    dynamic_output.sum().backward()

    for static_param, dynamic_param in zip(static_model.parameters(), dynamic_model.parameters()):
        assert compare_tensors(static_param.grad, dynamic_param.grad)



# device = xm.xla_device()
# model = MixtralForCausalLM(config).to(device)
# output = model(torch.randint(128, ((2, 128))).to(device))
# loss = torch.sum(output.logits)
# loss.backward()
# xm.mark_step()
# print(met.metrics_report())

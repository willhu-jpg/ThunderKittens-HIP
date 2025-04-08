import torch

B = 1
N = 16
D = 32

"""
Reference Logic: https://github.com/HazyResearch/ThunderKittens/blob/tk_gen/simple_kernels/micro_add/
"""

def add(x):
    """
    o = x + x
    """
    o = x + x
    # print("Result tensor:", result.mean().item(), "dtype:", result.dtype)  # Debug output
    return o

# input = torch.ones((B, N, D), dtype=torch.bfloat16, device='cuda')

input = torch.ones((B, N, D), dtype=torch.float32, device='cuda')
print("Input tensor:", input.mean().item(), "dtype:", input.dtype)  # Debug input

output_ref = add(input)
print("Ref output mean:", output_ref.mean().item())  # Debug final output

input_copy = input.clone()

output_tk = torch.zeros_like(input_copy)

import simple_tk
simple_tk.dispatch_micro(input_copy, output_tk)

print(output_tk)
print("TK output mean:", output_tk.mean().item())  # Debug final output

# Okay sofar this is wrong!
if torch.allclose(output_ref, output_tk):
    print("[Passed] TK Kernel matches reference")
else:
    print("[Failed] TK Kernel does not match reference")
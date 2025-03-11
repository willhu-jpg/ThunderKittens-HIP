import simple_tk


import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Reference Logic: https://github.com/HazyResearch/ThunderKittens/blob/tk_gen/simple_kernels/micro_add/


With TK Problems, we should also specify dtypes
"""

B = 1 
N = 16
D = 32
DTYPE = torch.float32


def get_inputs():
    # randomly generate input tensors based on the model architecture
    x = torch.randn(B, N, D, dtype=DTYPE).cuda()
    return [x]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []


########################################################
# Reference Model
########################################################

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        # o = x + x 
        return x + x

########################################################
# New Model
########################################################

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        output = torch.zeros_like(x, dtype=DTYPE)
        simple_tk.dispatch_micro(x, output)
        return output


def compare_models():
    """
    Compare the reference model and the new model (with TK kernels)
     """
    inputs = get_inputs()
    init_inputs = get_init_inputs()

    model_ref = Model(*init_inputs).cuda()
    model_new = ModelNew(*init_inputs).cuda()


    output_ref = model_ref.forward(*inputs)
    print("Ref output mean:", output_ref.mean().item())

    output_tk = model_new.forward(*inputs)
    print("New output mean:", output_tk.mean().item())

    if torch.allclose(output_ref, output_tk):
        print("[Passed] TK Kernel matches reference")
    else:
        print("[Failed] TK Kernel does not match reference")


if __name__ == "__main__":
    compare_models()
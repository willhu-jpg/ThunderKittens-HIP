import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import sys
import math

B = 1
N = 16
D = 32

TESTNAME = sys.argv[1]

print("torch.version.hip:", torch.version.hip)
if hasattr(torch.backends, "hip"):
    print("ROCm available:", torch.backends.hip.is_available())
else:
    print("This PyTorch build does not have ROCm support.")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("Device name:", props.name)

if TESTNAME == 'ones':
    x = torch.ones((B, N, D), dtype=torch.float32, device='cuda')
elif TESTNAME == 'randn':
    x = torch.randn((B, N, D), dtype=torch.float32, device='cuda') / D
elif TESTNAME == "arange":
    x = torch.arange(B*N*D, dtype=torch.float32, device='cuda').reshape(B, N, D)
else:
    print('Invalid test name')
    sys.exit(0)

def get_output(x):
    # TEST 1: add
    o = x + x
    return o

o = get_output(x)

with open(f'{TESTNAME}.txt', 'w') as f:
    xf = x.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    of = o.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    for i in trange(len(xf)):
        f.write(repr(xf[i]))
        f.write(' ')
    for i in trange(len(of)):
        f.write(repr(of[i]))
        f.write(' ')
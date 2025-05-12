import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import sys
import math

M = 16
N = 16
K = 16

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
    x = torch.ones((M, K), dtype=torch.bfloat16, device='cuda')
    y = torch.ones((K, N), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'randn':
    x = torch.randn((M, K), dtype=torch.bfloat16, device='cuda')
    y = torch.randn((K, N), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == "arange":
    x = torch.arange(M*K, dtype=torch.bfloat16, device='cuda').reshape(M, K)
    y = torch.arange(K*N, dtype=torch.bfloat16, device='cuda').reshape(K, N)
else:
    print('Invalid test name')
    sys.exit(0)

def get_output(x, y):
    # TEST 1: add
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        o = torch.matmul(x, y)
    return o

o = get_output(x, y)

with open(f'{TESTNAME}.txt', 'w') as f:
    xf = x.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    yf = y.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    of = o.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    for i in trange(len(xf)):
        f.write(repr(xf[i]))
        f.write(' ')
    for i in trange(len(yf)):
        f.write(repr(yf[i]))
        f.write(' ')
    for i in trange(len(of)):
        f.write(repr(of[i]))
        f.write(' ')
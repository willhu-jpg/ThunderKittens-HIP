import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import numpy as np
import sys
import math

# 16x32 * 32x16
batch_size = 16
in_channels = 32
hidden_channels = 16

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
    data = torch.ones((batch_size, in_channels), dtype=torch.bfloat16, device='cuda')

    W1 = torch.ones((in_channels, hidden_channels), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'randn':
    data = torch.randn((batch_size, in_channels), dtype=torch.bfloat16, device='cuda')

    W1 = torch.randn((in_channels, hidden_channels), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == "arange":
    data = torch.arange(batch_size * in_channels, dtype=torch.bfloat16, device='cuda').reshape(batch_size, in_channels)

    W1 = torch.arange(in_channels * hidden_channels, dtype=torch.bfloat16, device='cuda').reshape(in_channels, hidden_channels)
else:
    print('Invalid test name')
    sys.exit(0)

def get_output(data, W1):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        o = F.relu(torch.matmul(data, W1))
    return o

o = get_output(data, W1)

with open(f'{TESTNAME}.txt', 'w') as f:
    dataf = data.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    W1f = W1.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    of = o.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    for i in trange(len(dataf)):
        f.write(repr(dataf[i]))
        f.write(' ')
    for i in trange(len(W1f)):
        f.write(repr(W1f[i]))
        f.write(' ')
    for i in trange(len(of)):
        f.write(repr(of[i]))
        f.write(' ')
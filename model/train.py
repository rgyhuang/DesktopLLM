#!/usr/bin/env python
"""
This file includes functions and routines relevant to training/finetuning
the GPT model defined in model.py.
"""

import torch
from utils import DataLoader
from model import GPT, GPTConfig
import time

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # get data batch
    # enc = tiktoken.get_encoding("gpt2")
    # with open("input.txt", "r") as f:
    #     text = f.read()
    # data = text[:1000]
    # tokens = enc.encode(data)
    B, T = 8, 512
    # buf = torch.Tensor(tokens[:B * T + 1]).to(device).long()
    # x = buf[:-1].view(B, T)
    # y = buf[1:].view(B, T)

    model = GPT(GPTConfig())
    model.to(device)

    # compilation does not currently work with mps backend (missing ops)
    if device == "cuda":
        model = torch.compile(model)

    # logits, loss = model(x, y)
    # print(loss)
    trainLoader = DataLoader(B, T, device=device)

    # treat multiplcations using TF32 or bfloat16
    torch.set_float32_matmul_precision("high")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 50
    for i in range(epochs):
        t0 = time.time()
        x, y = next(trainLoader)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # lower precision for higher perfomance in nvidia ampere architecture
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss.backward()
        optimizer.step()

        # synchronize for accurate timing
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        print(
            f"Epoch {i}: loss = {loss.item()}, time = {(time.time() - t0) * 1000:0.2f} ms "
        )

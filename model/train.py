#!/usr/bin/env python
"""
This file includes functions and routines relevant to training/finetuning
the GPT model defined in model.py.
"""

import torch
from utils import DataLoader
from model import GPT, GPTConfig

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
    B, T = 4, 32
    # buf = torch.Tensor(tokens[:B * T + 1]).to(device).long()
    # x = buf[:-1].view(B, T)
    # y = buf[1:].view(B, T)

    model = GPT(GPTConfig())
    model.to(device)
    # logits, loss = model(x, y)
    # print(loss)
    trainLoader = DataLoader(B, T, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 50
    for i in range(epochs):
        x, y = next(trainLoader)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {i}: loss = {loss.item()}")

    # num_return_sequences = 5
    # max_sequence_length = 30

    # prefix token init
    # enc = tiktoken.get_encoding("gpt2")
    # tokens = enc.encode("Hello, I am a language model")
    # tokens = torch.tensor(tokens, dtype=torch.long)
    # tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    # x = tokens.to("mps")

    # start_time = time.time()
    # # generate using model
    # while x.shape[1] < max_sequence_length:
    #     # forward the model to get logits
    #     with torch.no_grad():
    #         logits = model(x) # (num_return_sequences, max_sequence_length, vocab_size)

    #         logits = logits[:, -1, :] # (num_return_sequences, vocab_size)

    #         probs = F.softmax(logits, dim=-1) # (num_return_sequences, vocab_size)

    #         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

    #         ix = torch.multinomial(topk_probs, num_samples=1) # (num_return_sequences, 1)

    #         xcol = torch.gather(topk_indices, -1, ix) # (num_return_sequences, 1)

    #         x = torch.concat((x, xcol), dim=-1) # (num_return_sequences, max_sequence_length)

    # print(f"{5} phrases generated in {time.time() - start_time:.2f} seconds")

    # for i in range(num_return_sequences):
    #     gen_tokens = x[i, :max_sequence_length].tolist()
    #     decoded = enc.decode(gen_tokens)
    #     print("> ", decoded)

import tiktoken
import torch


class DataLoader:
    def __init__(self, B, T, device="cpu"):
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens).to(device)

        print(f"Tokens loaded: {len(self.tokens)}")
        print(f"Batches per epoch: {len(self.tokens) // (B * T)}")

        self.current_position = 0

    def __next__(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T

        if self.current_position >= len(self.tokens) - 1:
            self.current_position = 0
        return x, y

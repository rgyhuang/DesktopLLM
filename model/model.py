#!/usr/bin/env python
"""
This file implements a simple transformer-based GPT model, adapted from
the paper 'Attention Is All You Need' (Vaswani et al., 2017) and
'Language Models are Unsupervised Multitask Learners' (Radford et al., 2019).
"""

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


# Masked multi-head attention
class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention, evaluating
    total_heads attention heads in parallel
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.attn_dropout = nn.Dropout(config.p_drop)
        self.resid_dropout = nn.Dropout(config.p_drop)

    def forward(self, x):
        # batch, time, channels
        B, T, C = x.size()
        qkv = self.c_attn(x)
        Q, K, V = qkv.split(self.n_embd, dim=2)
        hs = C // self.n_head
        # (B, T, hs * n_head) -> (B, n_head, T, hs)
        Q = Q.view(B, T, self.n_head, hs).transpose(1, 2)
        K = K.view(B, T, self.n_head, hs).transpose(1, 2)
        V = V.view(B, T, self.n_head, hs).transpose(1, 2)

        # use flash attention for better performance
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # reassemble all head outputs
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # project and dropout
        out = self.resid_dropout(self.c_proj(out))

        return out


class FeedForward(nn.Module):
    """
    Implements a simple feed forward network
    consisting of two linear transformations with
    a ReLU activation in between
    """

    def __init__(self, n_embd, dropout) -> None:
        super().__init__()
        # n_embd * 4 in the original paper
        # GeLU is smoother than ReLU, can improve performance in transformer (i.e. dead ReLU)
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.c_proj.SCALE_INIT = 1

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        out = self.dropout(x)

        return out


class Transformer(nn.Module):
    """
    Implements a transformer block using multi-head attention, feed forward, and add/norm
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.mlp = FeedForward(config.n_embd, config.p_drop)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # layer_norm was moved to the input of each sub-block in GPT2
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max context length for predictions
    vocab_size: int = 50257
    n_layer: int = 12  # number of transformer blocks
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension
    p_drop: float = 0.1  # dropout probability


class GPT(nn.Module):
    """
    Implements a simple transformer-based GPT model
    """

    def __init__(self, config):
        super().__init__()
        self.GPTConfig = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.GPTConfig.vocab_size, self.GPTConfig.n_embd),
                wpe=nn.Embedding(self.GPTConfig.block_size, self.GPTConfig.n_embd),
                h=nn.ModuleList(
                    [Transformer(self.GPTConfig) for _ in range(self.GPTConfig.n_layer)]
                ),
                # additional layer added in GPT2
                ln_f=nn.LayerNorm(self.GPTConfig.n_embd),
            )
        )
        self.lm_head = nn.Linear(
            self.GPTConfig.n_embd, self.GPTConfig.vocab_size, bias=False
        )

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # initialize weights
        self.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT") is not None:
                std *= (2 * self.GPTConfig.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None):
        # x is the batch of tokens
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_embd = self.transformer.wpe(pos)
        tok_embd = self.transformer.wte(x)
        x = pos_embd + tok_embd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        """
        Loads pretrained GPT-2 model weights from huggingface
        Adapted from Andrej Karpathy's nanogpt repo
        """

        assert pretrained_model_name_or_path in {
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
        }
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % pretrained_model_name_or_path)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[pretrained_model_name_or_path]

        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

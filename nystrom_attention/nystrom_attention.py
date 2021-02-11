from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

# helper functions

def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = x.transpose(-1, -2) / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)[None, ...]

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main class

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        m = 256,
        pinv_iterations = 6,
        residual = True
    ):
        super().__init__()
        inner_dim = heads * dim_head

        self.m = m
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.residual = residual
        if residual:
            self.res_conv = nn.Conv2d(heads, heads, 1, groups = heads, bias = False)

    def forward(self, x, mask = None):
        b, n, _, h, m, iters = *x.shape, self.heads, self.m, self.pinv_iterations

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, 0, padding), value = 0)

            if exists(mask):
                mask = F.pad(mask, (0, padding), value = False)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q *= self.scale

        l = ceil(q.shape[1] / m)
        q_landmarks = reduce(q, '... (n l) d -> ... n d', 'mean', l = l)
        k_landmarks = reduce(k, '... (n l) d -> ... n d', 'mean', l = l)

        if exists(mask):
            mask_landmarks = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            mask_landmarks = mask_landmarks > 0

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = attn1 @ attn2_inv @ attn3 @ v

        if self.residual:
            out += self.res_conv(v)

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        return out[:, :n]

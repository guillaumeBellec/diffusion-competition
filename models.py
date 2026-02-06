"""
Shared components for diffusion models.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_neighborhood_mask(grid_size=8, radius=1):
    """Build a binary attention mask for spatial neighborhood filtering.

    For an (grid_size x grid_size) spatial grid, returns a (N, N) boolean mask
    where mask[i, j] = True iff positions i and j are within `radius` steps
    (Chebyshev distance) of each other.
    """
    N = grid_size * grid_size
    coords = np.array(np.unravel_index(np.arange(N), (grid_size, grid_size))).T  # (N, 2)
    # Chebyshev distance between all pairs
    diff = np.abs(coords[:, None, :] - coords[None, :, :])  # (N, N, 2)
    dist = diff.max(axis=-1)  # (N, N)
    return dist <= radius


class Diffusion:
    """DDPM/DDIM diffusion process."""

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # DDPM precomputed values
        self.betas = betas
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def diffuse(self, x0, t):
        """Add noise to x0 at timestep t."""
        noise = torch.randn_like(x0)
        xt = (self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x0 +
              self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise)
        return xt, noise

    def ddpm_step(self, xt, noise_pred, t):
        """DDPM stochastic step from t to t-1."""
        coef = self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t]
        mean = self.sqrt_recip_alphas[t] * (xt - coef * noise_pred)
        if t > 0:
            return mean + torch.sqrt(self.posterior_variance[t]) * torch.randn_like(xt)
        return mean

    def ddim_step(self, xt, noise_pred, t, t_next):
        """DDIM deterministic step from t to t_next."""
        alpha_t = self.alphas_cumprod[t]
        alpha_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=xt.device)
        x0_pred = (xt - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        return alpha_next.sqrt() * x0_pred + (1 - alpha_next).sqrt() * noise_pred


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        from copy import deepcopy
        self.decay = decay
        self.shadow = deepcopy(model)
        for p in self.shadow.parameters():
            p.detach_()

    @torch.no_grad()
    def update(self, model):
        for sp, p in zip(self.shadow.parameters(), model.parameters()):
            sp.sub_((1 - self.decay) * (sp - p))
        for sb, b in zip(self.shadow.buffers(), model.buffers()):
            sb.copy_(b)


# ============ DiT Components ============

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for spatial grids."""

    def __init__(self, head_dim, grid_size):
        super().__init__()
        quarter = head_dim // 4
        freqs = 1.0 / (10000 ** (torch.arange(quarter).float() / quarter))
        ys, xs = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
        # x-frequencies for first half of pairs, y-frequencies for second half
        angles_x = xs.flatten().float()[:, None] * freqs[None, :]  # (N, quarter)
        angles_y = ys.flatten().float()[:, None] * freqs[None, :]  # (N, quarter)
        angles = torch.cat([angles_x, angles_y], dim=-1)  # (N, head_dim//2)
        self.register_buffer('cos', angles.cos()[None, None])  # (1, 1, N, head_dim//2)
        self.register_buffer('sin', angles.sin()[None, None])

    def forward(self, q, k):
        """Apply rotary embeddings to q and k. Shape: (B, heads, N, head_dim)."""
        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack([x1 * self.cos - x2 * self.sin,
                                x1 * self.sin + x2 * self.cos], dim=-1).flatten(-2)
        return rotate(q), rotate(k)


class Attention(nn.Module):
    def __init__(self, dim, heads, attn_mask=None, rope_2d_grid_size=None):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_mask = attn_mask  # optional (N, N) boolean tensor
        self.rope = RoPE2D(dim // heads, rope_2d_grid_size) if rope_2d_grid_size else None

    def forward(self, x):
        B, N, C = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.rope:
            q, k = self.rope(q, k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.cond_scale1 = nn.Parameter(torch.zeros(dim))
        self.cond_scale2 = nn.Parameter(torch.zeros(dim))
        self.skip_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x0, cond):
        x = x0 + self.cond_scale1 * cond
        x = x + self.attn(self.norm1(x))
        x = x + self.cond_scale2 * cond
        x = x + self.mlp(self.norm2(x))
        return x * self.skip_scale + (1 - self.skip_scale) * x0


# ============ UNet Components ============

class ResBlock(nn.Module):
    """Residual block with time/class conditioning via scale and shift (FiLM)."""

    def __init__(self, in_ch, out_ch, cond_dim, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)  # projects to scale and shift
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Self-attention for spatial features."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.unbind(1)
        q, k, v = q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

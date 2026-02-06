"""
Minimal Diffusion Model for CIFAR-10 Generation
Score-based diffusion with classifier-free guidance using a Transformer backbone.
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Diffusion, SinusoidalEmbedding, RMSNorm, EMA, Attention, make_neighborhood_mask
from torch.utils.checkpoint import checkpoint

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

class Config:
    # Model architecture
    dim = 6 * 48
    depth = 12
    heads = 6
    patch_size = 4
    img_size = 32
    num_classes = 10

    # Diffusion schedule
    diff_timesteps = 1000
    diff_beta_start = 1e-4
    diff_beta_end = 0.02
    diff_sample_steps = 100

    local_attn_dist = None # Neighborhood attention (None = full attention, int = Chebyshev radius)
    rope_2d = False # 2D Rotary Position Embedding (replaces learned pos_embed in attention)

    # Classifier-Free Guidance
    cfg_drop = 0.1
    cfg_scale = 2.0

    # Training
    batch_size = 512
    lr = 3e-4
    epochs = 300
    lr_warmup_steps = 1000
    ema_decay = 0.99

    device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def patch_dim(self):
        return self.patch_size ** 2 * 3 # num pixels x RGB


class ConfigMedium(Config):
    # Model architecture
    dim = 6 * 128
    depth = 12
    heads = 6

    # Training
    batch_size = 512 * 4
    lr = 1e-4
    epochs = 300 * 4
    lr_warmup_steps = 1000
    ema_decay = 0.99

class NCAConfig(Config):
    local_attn_dist = 2 # Neighborhood attention (None = full attention, int = Chebyshev radius)
    rope_2d = True # 2D Rotary Position Embedding (replaces learned pos_embed in attention)

class NCAConfigMedium(ConfigMedium):
    local_attn_dist = 2 # Neighborhood attention (None = full attention, int = Chebyshev radius)
    rope_2d = True # 2D Rotary Position Embedding (replaces learned pos_embed in attention)

class Block(nn.Module):
    def __init__(self, dim, heads, rope_2d_grid_size=None):
        super().__init__()
        self.attn = Attention(dim, heads, rope_2d_grid_size=rope_2d_grid_size)
        self.mlp = nn.Sequential(RMSNorm(dim),nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.cond_scale1 = nn.Parameter(torch.zeros(dim))
        self.cond_scale2 = nn.Parameter(torch.zeros(dim))
        self.skip_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x0, cond, attn_mask=None):
        x = x0
        x = x + self.cond_scale1 * cond
        x = x + self.attn(x, attn_mask=attn_mask)
        x = x + self.cond_scale2 * cond
        x = x + self.mlp(x)
        return x * self.skip_scale + (1-self.skip_scale) * x0


class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.diffusion = Diffusion(config.diff_timesteps, config.diff_beta_start, config.diff_beta_end, config.device)

        self.patch_embed = nn.Linear(config.patch_dim, config.dim)
        grid_size = config.img_size // config.patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size * grid_size, config.dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.input_norm = RMSNorm(config.dim)
        self.time_embed = SinusoidalEmbedding(config.dim)
        self.class_embed = nn.Embedding(config.num_classes + 1, config.dim)

        # Build neighborhood attention mask if configured
        attn_mask = None
        if config.local_attn_dist is not None:
            attn_mask = torch.from_numpy(make_neighborhood_mask(grid_size, config.local_attn_dist))
        self.register_buffer('attn_mask', attn_mask, persistent=False)

        rope_gs = grid_size if getattr(config, 'rope_2d', False) else None
        self.blocks = nn.ModuleList([Block(config.dim, config.heads, rope_2d_grid_size=rope_gs) for _ in range(config.depth)])
        self.norm = RMSNorm(config.dim)
        self.out = nn.Linear(config.dim, config.patch_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def patchify(self, x):
        p = self.config.patch_size
        return x.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).flatten(1, 2).flatten(2)

    def unpatchify(self, x):
        p, s = self.config.patch_size, self.config.img_size
        g = s // p
        return x.reshape(-1, g, g, 3, p, p).permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, s, s)

    def forward(self, x_in, t, c):
        x = self.patch_embed(self.patchify(x_in))
        x = self.input_norm(x) + self.pos_embed
        t_emb = self.time_embed(t)  # [B, dim]
        c_emb = self.class_embed(c)  # [B, dim]
        cond = (t_emb + c_emb).unsqueeze(1)  # [B, 1, dim]
        for block in self.blocks:
            x = block(x, cond, attn_mask=self.attn_mask)
        x_pred = self.unpatchify(self.out(self.norm(x)))

        return x_pred

    @torch.no_grad()
    def sample(self, class_ids, n_steps=None, guidance_scale=None):
        """Sample images using DDIM with Classifier-Free Guidance."""
        config = self.config
        n_steps = n_steps or config.diff_sample_steps
        guidance_scale = guidance_scale if guidance_scale is not None else config.cfg_scale
        B, device = len(class_ids), class_ids.device
        null_c = torch.full_like(class_ids, config.num_classes)

        # Timesteps: T-1 -> 0 with n_steps+1 points (to have n_steps intervals)
        steps = torch.linspace(config.diff_timesteps - 1, 0, n_steps + 1, dtype=torch.long, device=device)

        # Init with pure noise
        x = torch.randn(B, 3, config.img_size, config.img_size, device=device)

        # DDIM sampling
        for i in range(n_steps):
            t, t_next = steps[i], steps[i + 1]
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)

            # Predict noise with CFG
            eps_cond = self.forward(x, t_batch, class_ids)
            eps_uncond = self.forward(x, t_batch, null_c)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            x = self.diffusion.ddim_step(x, eps, t, t_next)

        return x


def train(config=None):
    import matplotlib.pyplot as plt

    config = config or ConfigMedium()
    print(f"Training on {config.device} for {config.epochs} epochs")

    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    loader = DataLoader(
        datasets.CIFAR10("./data", train=True, download=True, transform=tf),
        batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
        persistent_workers=True
    )

    model = DiT(config).to(config.device)
    #model = torch.compile(model)

    norm_params = [p for n, p in model.named_parameters() if 'scale' in n or 'norm' in n]
    embed_params = [p for n, p in model.named_parameters() if 'embed' in n]
    other_params = [p for n, p in model.named_parameters()
                    if not any(x in n for x in ['scale', 'norm', 'embed'])]

    n_norm = sum(p.numel() for p in norm_params) // 1000
    n_embed = sum(p.numel() for p in embed_params) // 1000
    n_other = sum(p.numel() for p in other_params) // 1000
    print(f"Parameters (k): norm={n_norm} embed={n_embed} other={n_other} total={n_norm+n_embed+n_other}")

    opt = torch.optim.AdamW([
        {'params': other_params, 'lr': config.lr},
        {'params': norm_params, 'lr': config.lr * 5},
        {'params': embed_params, 'lr': config.lr * 2},
    ], weight_decay=0.01, fused=True)

    #ema = EMA(model, config.ema_decay)

    def lr_lambda(step):
        if step < config.lr_warmup_steps:
            return step / config.lr_warmup_steps
        return 1.

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    plt.ion()
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    plt.show(block=False)

    step = 0
    loss_history = []
    for epoch in range(config.epochs):
        epoch_start = time.time()
        model.train()
        losses = []

        for imgs, labels in loader:
            imgs = imgs.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)

            t = torch.randint(0, config.diff_timesteps, (imgs.shape[0],), device=config.device)

            # Classifier-Free Guidance: randomly drop class labels during training
            drop = torch.rand(labels.shape[0], device=config.device) < config.cfg_drop
            labels = torch.where(drop, config.num_classes, labels)

            # forward pass where we get the noisy image and the noise
            xt, noise = model.diffusion.diffuse(imgs, t)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                noise_pred = model(xt, t, labels)
                loss = F.mse_loss(noise_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            #ema.update(model)

            losses.append(loss.item())
            step += 1

        avg_loss = sum(losses) / len(losses)
        loss_history.append(avg_loss)
        train_time = time.time() - epoch_start

        # Show samples every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            sample_classes = torch.tensor([0, 1, 2, 3, 4, 5], device=config.device)
            with torch.no_grad():
                samples = model.sample(sample_classes)
            samples = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()

            # Show 5 images in first 5 subplots
            for idx in range(5):
                ax = axes.flat[idx]
                ax.clear()
                img = np.transpose(samples[idx], (1, 2, 0))
                ax.imshow(img)
                ax.set_title(f"{class_names[sample_classes[idx]]}")
                ax.axis("off")

            # Loss curve in last subplot
            ax = axes.flat[5]
            ax.clear()
            ax.plot(loss_history, 'b-', linewidth=1)
            ax.set_ylim([0.01,0.06])
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_title('loss')
            ax.grid(True, alpha=0.3)

            fig.suptitle(f"Epoch {epoch + 1}/{config.epochs} | Loss: {avg_loss:.4f}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1)

        print(f"Epoch {epoch + 1:3d}/{config.epochs} | Loss: {avg_loss:.4f} | Time: {train_time:.1f}s")


    #ema.apply(model)
    torch.save({
        "model": model.state_dict(),
        "config_name": type(config).__name__,
        "config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
    }, f"diffusion_cifar10_{type(config).__name__}.pth")
    print(f"Saved: diffusion_cifar10_{type(config).__name__}.pth")

    plt.ioff()
    plt.close(fig)
    return model


class Agent:
    def __init__(self):
        checkpoint = torch.load("diffusion_cifar10.pth", map_location="cpu", weights_only=False)
        self.config = Config() # load default will be overloaded
        for k, v in checkpoint["config"].items():
            setattr(self.config, k, v)
        self.config.device = "cpu"
        self.model = DiT(self.config).to(self.config.device)
        # Strip _orig_mod. prefix from compiled model checkpoint
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()


    def generate(self, class_ids: np.ndarray) -> np.ndarray:
        c = torch.from_numpy(class_ids).long().to(self.config.device)
        with torch.no_grad():
            imgs = self.model.sample(c)
        imgs = ((imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        return imgs.cpu().numpy()


if __name__ == "__main__":
    train(Config())
    train(ConfigMedium())
    train(NCAConfig())
    train(NCAConfigMedium())

"""
UNet Diffusion Model for CIFAR-10 Generation
Score-based diffusion with classifier-free guidance using a UNet backbone.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Diffusion, SinusoidalEmbedding, EMA, ResBlock, Downsample, Upsample, SelfAttention2d

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


class Config:
    # Model
    base_dim = 64       # channels: C, 2C, 4C at resolutions 32, 16, 8
    num_classes = 10

    # Diffusion
    diff_timesteps = 1000
    diff_beta_start = 1e-4
    diff_beta_end = 0.02
    diff_sample_steps = 100

    # Classifier-free guidance
    cfg_drop = 0.1
    cfg_scale = 2.0

    # Training
    batch_size = 512
    lr = 2e-4
    epochs = 300
    lr_warmup_steps = 1000
    ema_decay = 0.999

    device = "cuda" if torch.cuda.is_available() else "cpu"


class UNet(nn.Module):
    """
    UNet with time and class conditioning.
    Architecture: 32x32 -> 16x16 -> 8x8 -> 16x16 -> 32x32 with skip connections.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.diffusion = Diffusion(
            config.diff_timesteps, config.diff_beta_start, config.diff_beta_end, config.device
        )
        C = config.base_dim  # base channels
        cond_dim = C * 4

        # Conditioning embeddings
        self.time_embed = SinusoidalEmbedding(cond_dim)
        self.class_embed = nn.Embedding(config.num_classes + 1, cond_dim)

        # Encoder path: 32x32 -> 16x16 -> 8x8
        self.conv_in = nn.Conv2d(3, C, 3, padding=1)
        self.enc1 = ResBlock(C, C, cond_dim)          # 32x32
        self.down1 = Downsample(C)                     # -> 16x16
        self.enc2 = ResBlock(C, C * 2, cond_dim)      # 16x16
        self.down2 = Downsample(C * 2)                 # -> 8x8
        self.enc3 = ResBlock(C * 2, C * 4, cond_dim)  # 8x8
        self.down3 = Downsample(C * 4)                 # -> 4x4
        self.enc4 = ResBlock(C * 4, C * 8, cond_dim)  # 4x4

        # Middle (bottleneck at 4x4)
        self.mid1 = ResBlock(C * 8, C * 8, cond_dim)
        self.mid_attn = SelfAttention2d(C * 8)
        self.mid2 = ResBlock(C * 8, C * 8, cond_dim)

        # Decoder path: 8x8 -> 16x16 -> 32x32 (with skip connections)
        self.dec4 = ResBlock(C * 8 + C * 8, C * 8, cond_dim)  # concat skip from enc3
        self.up3 = Upsample(C * 8)                             # -> 16x16
        self.dec3 = ResBlock(C * 8 + C * 4, C * 4, cond_dim)  # concat skip from enc3
        self.up2 = Upsample(C * 4)                             # -> 16x16
        self.dec2 = ResBlock(C * 4 + C * 2, C * 2, cond_dim)  # concat skip from enc2
        self.up1 = Upsample(C * 2)                             # -> 32x32
        self.dec1 = ResBlock(C * 2 + C, C, cond_dim)          # concat skip from enc1

        # Output
        self.norm_out = nn.GroupNorm(min(8, C), C)
        self.conv_out = nn.Conv2d(C, 3, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, t, c):
        cond = self.time_embed(t) + self.class_embed(c)

        # Encoder
        h = self.conv_in(x)                    # -> C x 32 x 32
        s1 = self.enc1(h, cond)                # -> C x 32 x 32
        s2 = self.enc2(self.down1(s1), cond)   # -> 2C x 16 x 16
        s3 = self.enc3(self.down2(s2), cond)   # -> 4C x 8 x 8
        s4 = self.enc4(self.down3(s3), cond)   # -> 8C x 4 x 4

        # Middle
        h = self.mid1(s4, cond)                 # -> 8C x 4 x 4
        h = self.mid_attn(h)
        h = self.mid2(h, cond)

        # Decoder with skip connections
        h = self.dec4(torch.cat([h, s4], 1), cond)   # -> 8C x 4 x 4
        h = self.dec3(torch.cat([self.up3(h), s3], 1), cond)   # -> 4C x 8 x 8
        h = self.dec2(torch.cat([self.up2(h), s2], 1), cond)  # -> 2C x 16 x 16
        h = self.dec1(torch.cat([self.up1(h), s1], 1), cond)  # -> C x 32 x 32

        return self.conv_out(F.silu(self.norm_out(h)))

    @torch.no_grad()
    def sample(self, class_ids, n_steps=None, guidance_scale=None):
        """Sample with DDIM and CFG."""
        cfg = self.config
        n_steps = n_steps or cfg.diff_sample_steps
        guidance_scale = guidance_scale if guidance_scale is not None else cfg.cfg_scale
        B, device = len(class_ids), class_ids.device
        null_c = torch.full_like(class_ids, cfg.num_classes)

        steps = torch.linspace(cfg.diff_timesteps - 1, 0, n_steps + 1, dtype=torch.long, device=device)
        x = torch.randn(B, 3, 32, 32, device=device)

        for i in range(n_steps):
            t, t_next = steps[i], steps[i + 1]
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)

            eps_cond = self.forward(x, t_batch, class_ids)
            eps_uncond = self.forward(x, t_batch, null_c)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            x = self.diffusion.ddim_step(x, eps, t, t_next)

        return x


def train(config=None):
    import matplotlib.pyplot as plt

    config = config or Config()
    print(f"Training UNet on {config.device} for {config.epochs} epochs")

    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    loader = DataLoader(
        datasets.CIFAR10("./data", train=True, download=True, transform=tf),
        batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True,
        drop_last=True, persistent_workers=True
    )

    model = UNet(config).to(config.device)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01, fused=True)
    ema = EMA(model, config.ema_decay)

    def lr_lambda(step):
        if step < config.lr_warmup_steps:
            return step / config.lr_warmup_steps
        return 1.0

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

            # CFG dropout
            drop = torch.rand(labels.shape[0], device=config.device) < config.cfg_drop
            labels = torch.where(drop, config.num_classes, labels)

            xt, noise = model.diffusion.diffuse(imgs, t)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                noise_pred = model(xt, t, labels)
                loss = F.mse_loss(noise_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ema.update(model)

            losses.append(loss.item())
            step += 1

        avg_loss = sum(losses) / len(losses)
        loss_history.append(avg_loss)
        train_time = time.time() - epoch_start

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            sample_classes = torch.tensor([0, 1, 2, 3, 4, 5], device=config.device)
            with torch.no_grad():
                samples = model.sample(sample_classes)
            samples = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()

            for idx in range(5):
                ax = axes.flat[idx]
                ax.clear()
                ax.imshow(np.transpose(samples[idx], (1, 2, 0)))
                ax.set_title(f"{class_names[sample_classes[idx]]}")
                ax.axis("off")

            ax = axes.flat[5]
            ax.clear()
            ax.plot(loss_history, 'b-', linewidth=1)
            ax.set_ylim([0.01, 0.06])
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_title('loss')
            ax.grid(True, alpha=0.3)

            fig.suptitle(f"Epoch {epoch + 1}/{config.epochs} | Loss: {avg_loss:.4f}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1)

        print(f"Epoch {epoch + 1:3d}/{config.epochs} | Loss: {avg_loss:.4f} | Time: {train_time:.1f}s")

    torch.save({
        "model": ema.shadow.state_dict(),
        "config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
    }, "diffusion_unet_cifar10.pth")
    print("Saved: diffusion_unet_cifar10.pth")

    plt.ioff()
    plt.close(fig)
    return model


class Agent:
    def __init__(self, model_path="diffusion_unet_cifar10.pth"):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.config = Config()
        for k, v in checkpoint["config"].items():
            setattr(self.config, k, v)
        self.config.device = "cpu"
        self.model = UNet(self.config).to(self.config.device)
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
    train()

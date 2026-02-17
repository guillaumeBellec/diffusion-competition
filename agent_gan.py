"""
Minimal GAN for CIFAR-10 Generation
Class-conditional GAN with ConvNet generator and discriminator.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.backends.cudnn.benchmark = True


class Config:
    img_size = 32
    num_classes = 10
    z_dim = 128        # latent noise dimension
    g_dim = 128        # generator base channels
    d_dim = 128        # discriminator base channels

    # Training
    batch_size = 512
    lr_g = 2e-4
    lr_d = 2e-4
    epochs = 300
    n_critic = 1       # discriminator steps per generator step

    device = "cuda" if torch.cuda.is_available() else "cpu"


# ============ Generator ============

class GenBlock(nn.Module):
    """Upsample + conv + batchnorm + relu."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        g = config.g_dim
        self.class_embed = nn.Embedding(config.num_classes, config.z_dim)

        # z + class_embed -> 4x4 feature map
        self.proj = nn.Sequential(
            nn.Linear(config.z_dim * 2, g * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        # 4x4 -> 8x8 -> 16x16 -> 32x32
        self.blocks = nn.Sequential(
            GenBlock(g * 8, g * 4),   # 4 -> 8
            GenBlock(g * 4, g * 2),   # 8 -> 16
            GenBlock(g * 2, g),       # 16 -> 32
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(g, 3, 3, padding=1),
            nn.Tanh(),
        )
        self.g_dim = g

    def forward(self, z, class_ids):
        c = self.class_embed(class_ids)
        h = self.proj(torch.cat([z, c], dim=1))
        h = h.view(-1, self.g_dim * 8, 4, 4)
        h = self.blocks(h)
        return self.to_rgb(h)


# ============ Discriminator ============

class DisBlock(nn.Module):
    """Conv + batchnorm + leaky relu + downsample."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_dim
        self.class_embed = nn.Embedding(config.num_classes, config.img_size * config.img_size)

        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.blocks = nn.Sequential(
            DisBlock(3 + 1, d),       # +1 for class channel, 32 -> 16
            DisBlock(d, d * 2),       # 16 -> 8
            DisBlock(d * 2, d * 4),   # 8 -> 4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d * 4 * 4 * 4, 1),
        )

    def forward(self, x, class_ids):
        # Inject class as a spatial channel
        c = self.class_embed(class_ids).view(-1, 1, x.shape[2], x.shape[3])
        h = torch.cat([x, c], dim=1)
        h = self.blocks(h)
        return self.head(h)


# ============ Training ============

def train(config=None):
    import matplotlib.pyplot as plt

    config = config or Config()
    print(f"Training on {config.device} for {config.epochs} epochs")

    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    loader = DataLoader(
        datasets.CIFAR10("./data", train=True, download=True, transform=tf),
        batch_size=config.batch_size, shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True, persistent_workers=True,
    )

    G = Generator(config).to(config.device)
    D = Discriminator(config).to(config.device)
    n_params_g = sum(p.numel() for p in G.parameters()) // 1000
    n_params_d = sum(p.numel() for p in D.parameters()) // 1000
    print(f"Parameters (k): G={n_params_g} D={n_params_d} total={n_params_g + n_params_d}")

    opt_g = torch.optim.Adam(G.parameters(), lr=config.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=config.lr_d, betas=(0.5, 0.999))

    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    plt.ion()
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    plt.show(block=False)

    loss_g_history, loss_d_history = [], []

    for epoch in range(config.epochs):
        epoch_start = time.time()
        G.train(); D.train()
        losses_g, losses_d = [], []

        for real_imgs, labels in loader:
            real_imgs = real_imgs.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)
            B = real_imgs.shape[0]

            # --- Train discriminator ---
            z = torch.randn(B, config.z_dim, device=config.device)
            with torch.no_grad():
                fake_imgs = G(z, labels)

            d_real = D(real_imgs, labels)
            d_fake = D(fake_imgs, labels)
            loss_d = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()  # hinge loss

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            losses_d.append(loss_d.item())

            # --- Train generator ---
            if len(losses_d) % config.n_critic == 0:
                z = torch.randn(B, config.z_dim, device=config.device)
                fake_imgs = G(z, labels)
                loss_g = -D(fake_imgs, labels).mean()

                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()
                losses_g.append(loss_g.item())

        avg_g = sum(losses_g) / len(losses_g) if losses_g else 0
        avg_d = sum(losses_d) / len(losses_d)
        loss_g_history.append(avg_g)
        loss_d_history.append(avg_d)
        train_time = time.time() - epoch_start

        # Show samples every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            G.eval()
            sample_classes = torch.tensor([0, 1, 2, 3, 4, 5], device=config.device)
            z = torch.randn(6, config.z_dim, device=config.device)
            with torch.no_grad():
                samples = G(z, sample_classes)
            samples = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()

            for idx in range(5):
                ax = axes.flat[idx]
                ax.clear()
                img = np.transpose(samples[idx], (1, 2, 0))
                ax.imshow(img)
                ax.set_title(f"{class_names[sample_classes[idx]]}")
                ax.axis("off")

            # Loss curves in last subplot
            ax = axes.flat[5]
            ax.clear()
            ax.plot(loss_g_history, 'b-', linewidth=1, label='G')
            ax.plot(loss_d_history, 'r-', linewidth=1, label='D')
            ax.legend(fontsize=8)
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_title('losses')
            ax.grid(True, alpha=0.3)

            fig.suptitle(f"Epoch {epoch + 1}/{config.epochs} | G: {avg_g:.4f} | D: {avg_d:.4f}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1)

        print(f"Epoch {epoch + 1:3d}/{config.epochs} | G: {avg_g:.4f} | D: {avg_d:.4f} | Time: {train_time:.1f}s")

    torch.save({"G": G.state_dict(), "config": vars(config)}, "gan_cifar10.pth")
    print("Saved: gan_cifar10.pth")

    plt.ioff()
    plt.close(fig)
    return G


# ============ Agent ============

class Agent:
    def __init__(self, model_path="gan_cifar10.pth"):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.config = Config()
        for k, v in checkpoint["config"].items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.config.device = "cpu"

        self.G = Generator(self.config)
        self.G.load_state_dict(checkpoint["G"])
        self.G.eval()

    def generate(self, class_ids: np.ndarray) -> np.ndarray:
        c = torch.from_numpy(class_ids).long()
        z = torch.randn(len(class_ids), self.config.z_dim)
        with torch.no_grad():
            imgs = self.G(z, c)
        imgs = ((imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        return imgs.cpu().numpy()


if __name__ == "__main__":
    train()

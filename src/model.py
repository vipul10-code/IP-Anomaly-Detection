"""
model.py
--------
A convolutional autoencoder (CAE) for unsupervised anomaly detection.

Why an autoencoder for v1 instead of a plain classifier CNN:
MVTec-style anomaly detection only gives you defect-free images at train time,
so a supervised classifier has nothing to learn "defect" from. An autoencoder
is trained to reconstruct only normal images; at test time it will reconstruct
normal regions well and anomalous regions poorly, and the per-pixel
reconstruction error map becomes the anomaly heatmap. This is the standard
first-version baseline before moving to feature-embedding methods like
PatchCore (see README "Roadmap").

Architecture: symmetric encoder-decoder, strided convs for downsampling,
transposed convs for upsampling, BatchNorm + LeakyReLU, sigmoid output.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, activation="leaky_relu"):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
        ]
        if activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvAutoencoder(nn.Module):
    """
    Input:  (B, 3, img_size, img_size), img_size must be a multiple of 32
             (default 256 -> 5 downsampling stages -> 8x8 bottleneck map).
    Output: (B, 3, img_size, img_size) reconstruction in [0, 1].
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, latent_channels: int = 100):
        super().__init__()
        c = base_channels

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, c),        # /2
            ConvBlock(c, c * 2),               # /4
            ConvBlock(c * 2, c * 4),           # /8
            ConvBlock(c * 4, c * 8),           # /16
            ConvBlock(c * 8, latent_channels, stride=2),  # /32  -> bottleneck
        )

        self.decoder = nn.Sequential(
            DeconvBlock(latent_channels, c * 8),
            DeconvBlock(c * 8, c * 4),
            DeconvBlock(c * 4, c * 2),
            DeconvBlock(c * 2, c),
            nn.ConvTranspose2d(c, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


def build_model(cfg: dict) -> ConvAutoencoder:
    m = cfg["model"]
    d = cfg["data"]
    return ConvAutoencoder(
        in_channels=d["channels"],
        base_channels=m["base_channels"],
        latent_channels=m["latent_channels"],
    )


if __name__ == "__main__":
    # quick shape sanity check
    net = ConvAutoencoder()
    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    print("input", x.shape, "-> output", y.shape)
    assert y.shape == x.shape

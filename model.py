"""Two small encoders, one per modality, plus the LeJEPA-style objective (invariance + SIGReg).

The image encoder is written so that `firmware/prh_model.c` can reproduce it exactly: four 3x3
stride-2 convolutions, ReLU, global average pool, linear. No normalisation layers, no residuals,
no attention — nothing a 240 MHz MCU cannot do.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 96
IMG_CH = [3, 24, 48, 64]   # three stride-2 convs: 96 -> 48 -> 24 -> 12
D_IMG = 64    # must not exceed the pooled feature width, or SIGReg has no attainable optimum
D_TXT = 256


class ImageEncoder(nn.Module):
    """96x96x3 -> D_IMG. Mirrors prh_model.c one-to-one."""

    def __init__(self, d_out=D_IMG, ch=IMG_CH):
        super().__init__()
        self.ch = ch
        self.convs = nn.ModuleList(
            nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1) for i in range(len(ch) - 1)
        )
        self.fc = nn.Linear(ch[-1], d_out)

    def forward(self, x):
        # x: [B, 3, 96, 96] in [0, 255]
        x = x / 128.0 - 1.0
        for c in self.convs:
            x = F.relu(c(x))
        x = x.mean(dim=(2, 3))
        return self.fc(x)


class TextEncoder(nn.Module):
    """Token ids -> D_TXT (mean pooling). Only used offline; never runs on the device."""

    def __init__(self, vocab, d_out=D_TXT, d_emb=64):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, d_emb)
        self.fc1 = nn.Linear(d_emb, 128)
        self.fc2 = nn.Linear(128, d_out)

    def forward(self, ids):
        e = self.emb(ids)
        mask = (ids < self.emb.num_embeddings - 1).unsqueeze(-1).float()
        h = (e * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc2(F.relu(self.fc1(h)))


def sigreg(x, knots=17, slices=256):
    """SIGReg: push the embedding distribution to N(0, I) through random 1-D projections.

    For each random unit direction u, the empirical characteristic function of x.u is compared to
    the N(0,1) characteristic function exp(-t^2/2) on a fixed grid, with the Epps-Pulley weight.
    """
    b, d = x.shape
    t = torch.linspace(0, 5, knots, device=x.device, dtype=x.dtype)
    u = torch.randn(d, slices, device=x.device, dtype=x.dtype)
    u = u / u.norm(dim=0, keepdim=True)
    z = x @ u                                            # [B, S]
    tx = z.unsqueeze(-1) * t                             # [B, S, T]
    cos = torch.cos(tx).mean(0)                          # [S, T]
    sin = torch.sin(tx).mean(0)
    target = torch.exp(-0.5 * t * t)
    w = torch.exp(-0.5 * t * t)
    return (((cos - target) ** 2 + sin ** 2) * w).sum(-1).mean()


def jepa_loss(h1, h2, mode="sigreg", inv_w=25.0, iso_w=0.1):
    """The paper's objective in miniature: the two views must agree (invariance) and the embedding
    must stay spread out (SIGReg / variance hinge). The invariance weight has to dominate: with a
    weight of 1 the encoder settles for a nearly constant output and only the isotropy term trains.

    mode="sigreg" (default, the paper's LeJEPA): SIGReg on RMS-normalised embeddings.
    mode="vicreg": variance hinge + covariance off-diagonals, kept as a comparison.
    """
    inv = F.mse_loss(h1, h2)
    h = torch.cat([h1, h2], 0)

    if mode == "vicreg":
        var = F.relu(1.0 - h.std(0)).mean()
        hc = h - h.mean(0)
        cov = (hc.T @ hc) / (len(h) - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        return inv_w * inv + inv_w * var + off.pow(2).sum() / len(cov), inv

    hn = h / (h.pow(2).mean().sqrt() + 1e-6)
    return inv_w * inv + iso_w * sigreg(hn), inv

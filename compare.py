"""Three arms, one architecture, one budget. The comparison the project exists to make.

    python compare.py --data trash --root /tmp/trash/dataset-resized --ckpt runs/trash.pt

Arm 1  supervised     the same CNN, trained end-to-end on labels with cross-entropy
Arm 2  prh zero-shot  the paper's method: two encoders trained separately, one rotation, classes
                      are sentences, no image ever seen with a label
Arm 3  prh + kNN      the aligned embedding plus 20 labelled photos per class (few-shot)

All three use `ImageEncoder` (43 k parameters) and the same number of optimisation steps.
"""

import argparse
import time

import numpy as np

import torch
import torch.nn.functional as F

import data_real, data_stl, data_trash
from model import ImageEncoder, TextEncoder

DATA = {"cifar": data_real, "stl": data_stl, "trash": data_trash}


def tensors(imgs, up, idx=None):
    x = torch.from_numpy(imgs if idx is None else imgs[idx]).permute(0, 3, 1, 2).float()
    if up > 1:
        x = x.repeat_interleave(up, 2).repeat_interleave(up, 3)
    return x


def train_supervised(D, root, n, steps, batch, dev, seed=0):
    torch.manual_seed(seed)
    imgs, _, labels, names = D.make_dataset(n, seed=seed, root=root)
    up = getattr(D, "UPSCALE", 1)
    xt, yt = D.test_set(root=root, n=1000, seed=seed)
    X = tensors(imgs[0], up)
    Y = torch.from_numpy(labels)
    Xt, Yt = tensors(xt, 1), torch.from_numpy(yt)

    enc = ImageEncoder().to(dev)
    head = torch.nn.Linear(enc.fc.out_features, len(names)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=3e-3)

    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, len(X), (batch,), generator=g)
        loss = F.cross_entropy(head(enc(X[idx].to(dev))), Y[idx].to(dev))
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        pred = head(enc(Xt.to(dev))).argmax(1).cpu()
    return float((pred == Yt).float().mean()), names


def prh_zero_shot(ck):
    return float(ck["acc"]), len(ck["names"])


def prh_knn(ck, D, root, shots=20, seed=0, k=5):
    """The aligned embedding, plus `shots` labelled photos per class, k-NN.

    Prototypes live in the *rotated* space, because that is the space the text side defined; a
    practitioner with a handful of photos would put them there too.
    """
    torch.manual_seed(seed)
    enc = ImageEncoder()
    enc.load_state_dict(ck["img"])
    enc.eval()
    up = getattr(D, "UPSCALE", 1)
    imgs, _, labels, names = D.make_dataset(2000, seed=seed + 5, root=root)
    xt, yt = D.test_set(root=root, n=1000, seed=seed)

    with torch.no_grad():
        R, mu_a, mu_b = ck["R"], ck["mu_a"], ck["mu_b"]
        move = lambda h: (h - mu_a) @ R + mu_b
        X = F.normalize(move(enc(tensors(imgs[0], up))), dim=-1)
        Xt = F.normalize(move(enc(tensors(xt, 1))), dim=-1)

    lab = np.asarray(labels)
    keep = [i for c in range(len(names)) for i in np.flatnonzero(lab == c)[:shots]]
    bank, bank_y = X[keep], torch.from_numpy(lab[keep])
    sim = Xt @ bank.T
    topk = sim.topk(k, dim=1).indices
    votes = torch.mode(bank_y[topk], dim=1).values
    return float((votes == torch.from_numpy(yt)).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=list(DATA), default="trash")
    ap.add_argument("--root", default="data/trash/dataset-resized")
    ap.add_argument("--ckpt", default="runs/trash.pt")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--shots", type=int, default=20)
    args = ap.parse_args()

    dev = torch.device("cpu")
    D = DATA[args.data]
    ck = torch.load(args.ckpt, weights_only=False)

    t0 = time.time()
    sup, names = train_supervised(D, args.root, args.n, args.steps, args.batch, dev)
    t_sup = time.time() - t0
    print(f"arm 1 supervised: {sup * 100:.1f} %", flush=True)

    zs, ncls = prh_zero_shot(ck)
    t0 = time.time()
    knn = prh_knn(ck, D, args.root, args.shots)
    t_knn = time.time() - t0
    print(f"arm 3 prh+knn: {knn * 100:.1f} %", flush=True)

    print(f"\ndata: {args.data} · {ncls} classes · {args.steps} steps · "
          f"{sum(p.numel() for p in ImageEncoder().parameters()):,} parameters\n")
    print("| arm | accuracy | labels used | train time |")
    print("|---|---|---|---|")
    print(f"| supervised, same CNN, end to end | {sup * 100:.1f} % | all | {t_sup / 60:.1f} min |")
    print(f"| paper's method, classes as sentences | {zs * 100:.1f} % | 0 image labels | "
          f"{ck.get('train_min', float('nan')):.1f} min |")
    print(f"| paper's method + {args.shots}-shot kNN | {knn * 100:.1f} % | "
          f"{args.shots * ncls} photos | {t_knn / 60:.1f} min |")
    print(f"\nchance: {100 / ncls:.1f} %")


if __name__ == "__main__":
    main()

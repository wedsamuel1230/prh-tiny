"""Is the 48 % a capacity problem or an alignment problem? One linear probe answers it.

The encoder is frozen. Only a linear classifier is trained, on labels, in the same 64-d space the
rotation starts from. That measures how much class information the encoder actually contains —
without any of the paper's machinery.

    probe ≈ supervised  -> the encoder is fine, the rotation/text side is the loss
    probe ≈ zero-shot   -> the encoder threw the information away, capacity is the loss
"""

import argparse

import torch
import torch.nn.functional as F

import data_real, data_stl, data_trash
from model import ImageEncoder

DATA = {"cifar": data_real, "stl": data_stl, "trash": data_trash}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="trash")
    ap.add_argument("--root", default="/tmp/trash/dataset-resized")
    ap.add_argument("--ckpt", default="runs/trash.pt")
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    D = DATA[args.data]
    ck = torch.load(args.ckpt, weights_only=False)
    enc = ImageEncoder()
    enc.load_state_dict(ck["img"])
    enc.eval()

    up = getattr(D, "UPSCALE", 1)
    imgs, _, labels, names = D.make_dataset(2500, seed=1, root=args.root)
    xt, yt = D.test_set(root=args.root, n=1000, seed=1)

    def batch(a, i=None):
        x = torch.from_numpy(a if i is None else a[i]).permute(0, 3, 1, 2).float()
        if up > 1:
            x = x.repeat_interleave(up, 2).repeat_interleave(up, 3)
        return x

    with torch.no_grad():
        H = enc(batch(imgs[0]))
        Ht = enc(batch(xt))
    H = F.normalize(H, dim=-1); Ht = F.normalize(Ht, dim=-1)
    Y = torch.from_numpy(labels); Yt = torch.from_numpy(yt)

    head = torch.nn.Linear(H.shape[1], len(names))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(args.steps):
        i = torch.randint(0, len(H), (128,), generator=g)
        loss = F.cross_entropy(head(H[i]), Y[i])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        probe = float((head(Ht).argmax(1) == Yt).float().mean())
        nn = (Ht @ H.T).topk(1, dim=1).indices.squeeze(1)
        knn1 = float((Y[nn] == Yt).float().mean())

    print(f"\nfrozen encoder, {H.shape[1]}-d, {len(names)} classes")
    print(f"  linear probe (labels, classifier only): {probe * 100:.1f} %")
    print(f"  1-NN over all 2,500 photos:             {knn1 * 100:.1f} %")
    print(f"  paper's method, zero-shot:              {ck['acc'] * 100:.1f} %")
    print(f"  chance:                                 {100 / len(names):.1f} %")


if __name__ == "__main__":
    main()

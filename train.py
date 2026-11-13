"""Train the two tiny encoders, fit the one rotation, measure the paper's claims.

    python train.py --views good --steps 600
    python train.py --views bad  --steps 600   # the control: corruptions of one observation

Nothing here is ESP32 specific; the device only ever runs ImageEncoder + Rotation.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import synth
from model import ImageEncoder, TextEncoder, jepa_loss, IMG_SIZE, D_IMG, D_TXT


def to_imgs(numpy_imgs, idx, dev="cpu", up=1):
    x = torch.from_numpy(numpy_imgs[idx]).permute(0, 3, 1, 2).float()
    if up > 1:
        x = x.repeat_interleave(up, 2).repeat_interleave(up, 3)
    return x.to(dev)


def to_ids(numpy_ids, idx, dev="cpu"):
    return torch.from_numpy(numpy_ids[idx]).long().to(dev)


def corrupt_images(rng, x, keep=0.6):
    """Bad views: two crops of ONE observation (a corruption), not two observations of the state."""
    b, _, h, w = x.shape
    out = []
    for _ in range(2):
        s = int(h * np.sqrt(keep))
        y0 = rng.integers(0, h - s + 1, size=b)
        x0 = rng.integers(0, w - s + 1, size=b)
        crops = torch.stack([x[i, :, y0[i]:y0[i] + s, x0[i]:x0[i] + s] for i in range(b)])
        out.append(F.interpolate(crops, size=(h, w), mode="bilinear", align_corners=False))
    return out


def corrupt_text(rng, ids, keep=0.6):
    """Bad views: two random spans of ONE caption (next-token prediction's constituents)."""
    out = []
    n = ids.shape[1]
    k = max(2, int(n * keep))
    for _ in range(2):
        start = rng.integers(0, n - k + 1, size=len(ids))
        m = torch.zeros_like(ids)
        for i in range(len(ids)):
            m[i, start[i]:start[i] + k] = ids[i, start[i]:start[i] + k]
        out.append(m)
    return out


def procrustes(a, b):
    u, _, vt = torch.linalg.svd(a.T @ b, full_matrices=False)
    return u @ vt


def whiten(x, eps=1e-2):
    c = x.T @ x / (len(x) - 1)
    c = c + eps * c.diagonal().mean() * torch.eye(len(c))
    e, v = torch.linalg.eigh(c)
    return x @ v @ torch.diag(e.clamp(min=1e-8).rsqrt()) @ v.T


def cca(a, b, top=8):
    """Mean of the top-`top` canonical correlations after whitening both spaces.

    The world here has ~8 effective dimensions, so the mean over all 128 directions would report
    a rank ratio, not the alignment. Everything above 1.0 now means a bug.
    """
    a = a - a.mean(0)
    b = b - b.mean(0)
    sv = torch.linalg.svdvals(whiten(a).T @ whiten(b) / (len(a) - 1))
    return float(sv[:top].mean()), float(sv.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", choices=["good", "bad"], default="good")
    ap.add_argument("--mode", choices=["sigreg", "vicreg"], default="sigreg")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/tiny.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", choices=["synth", "cifar", "stl", "trash"], default="synth")
    ap.add_argument("--root", default="cifar-10-batches-py")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if args.device == "auto" else args.device))
    print(f"device: {dev}")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    if args.data in ("cifar", "stl", "trash"):
        D = __import__({"cifar": "data_real", "stl": "data_stl", "trash": "data_trash"}[args.data])
        imgs, txts, labels, names = D.make_dataset(args.n, seed=args.seed, root=args.root)
        up = getattr(D, "UPSCALE", 1)
    else:
        import synth as D
        imgs, txts, labels, _ = D.make_dataset(args.n, seed=args.seed)
        names = None
        up = 1
    print(f"data: {args.data} {imgs.shape} images (x{up}), {txts.shape} captions  "
          f"({time.time() - t0:.1f}s)")

    img_enc = ImageEncoder().to(dev)
    txt_enc = TextEncoder(vocab=len(D.VOCAB)).to(dev)
    n_img = sum(p.numel() for p in img_enc.parameters())
    n_txt = sum(p.numel() for p in txt_enc.parameters())
    print(f"params: image {n_img:,} ({n_img / 1024:.0f} KiB int8)  text {n_txt:,} (offline)")

    o_img = torch.optim.AdamW(img_enc.parameters(), lr=args.lr)
    o_txt = torch.optim.AdamW(txt_enc.parameters(), lr=args.lr)

    # validation split for alignment: states the encoders never train on
    n_val = min(512, max(64, args.n // 5))
    tr = slice(0, args.n - n_val)
    va = slice(args.n - n_val, args.n)

    for step in range(1, args.steps + 1):
        idx = rng.integers(0, args.n - n_val, size=args.batch)
        x = to_imgs(imgs[0], idx, dev, up)
        if args.views == "good":
            v1, v2 = x, to_imgs(imgs[1], idx, dev, up)
        else:
            v1, v2 = corrupt_images(rng, x)

        h1, h2 = img_enc(v1), img_enc(v2)
        loss_i, inv_i = jepa_loss(h1, h2, mode=args.mode)
        o_img.zero_grad(); loss_i.backward(); o_img.step()

        t = to_ids(txts[0], idx, dev)
        if args.views == "good":
            w1, w2 = t, to_ids(txts[1], idx, dev)
        else:
            w1, w2 = corrupt_text(rng, t)

        g1, g2 = txt_enc(w1), txt_enc(w2)
        loss_t, inv_t = jepa_loss(g1, g2, mode=args.mode)
        o_txt.zero_grad(); loss_t.backward(); o_txt.step()

        if step % 100 == 0 or step == 1:
            print(f"step {step:4d}  img loss {loss_i.item():.4f} (inv {inv_i.item():.3f})"
                  f"  txt loss {loss_t.item():.4f} (inv {inv_t.item():.3f})")

    # ── the paper's claim: two encoders trained separately meet up to one rotation ──
    with torch.no_grad():
        a = img_enc(to_imgs(imgs[0], slice(va.start, va.stop), dev, up)).cpu()
        b = txt_enc(to_ids(txts[0], slice(va.start, va.stop), dev)).cpu()
        c, c_all = cca(a, b)
        mu_a, mu_b = a.mean(0), b.mean(0)
        R = procrustes(a - mu_a, b - mu_b)
        resid = float(((a - mu_a) @ R - (b - mu_b)).norm() / (b - mu_b).norm())

        # zero-shot: class prototypes come from captions only, never from images
        class_names = names if names is not None else [
            f"{synth.COLORS[ci % len(synth.COLORS)]} {synth.SHAPES[ci // len(synth.SHAPES)]}"
            for ci in range(len(synth.COLORS) * len(synth.SHAPES))]
        protos = []
        rng2 = np.random.default_rng(999)
        for ci, sents in enumerate(D.class_prototype_sentences(rng2)):
            protos.append(txt_enc(torch.tensor(sents)).mean(0))
        protos = F.normalize(torch.stack(protos), dim=-1)

        mapped = F.normalize((a - mu_a) @ R + mu_b, dim=-1)
        pred = (mapped @ protos.T).argmax(1)
        acc = float((pred == torch.from_numpy(labels[va.start:va.stop])).float().mean())

    print(f"\nalignment CCA top-8      {c:.3f}   (all-128 {c_all:.3f})")
    print(f"procrustes residual     {resid:.3f}")
    print(f"zero-shot top-1         {acc * 100:.1f}%  ({len(class_names)} classes, "
          f"chance {100 / len(class_names):.1f}%)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"img": img_enc.state_dict(), "txt": txt_enc.state_dict(),
                "R": R, "mu_a": mu_a, "mu_b": mu_b,
                "vocab": D.VOCAB, "names": class_names,
                "protos": protos, "views": args.views, "mode": args.mode,
                "cca": c, "cca_all": c_all, "data": args.data, "train_min": (time.time() - t0) / 60, "acc": acc, "n_img": n_img},
               args.out)
    print(f"saved {args.out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()

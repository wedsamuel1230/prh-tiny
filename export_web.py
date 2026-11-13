"""Export the trained model for the browser demo (site/model.json).

The image encoder and the rotation go out as int8 + per-channel scale, exactly like the firmware;
the text encoder goes out as float32, because it only ever runs on the host. The page dequantises
at load and runs the same arithmetic the ESP32 runs.
"""

import argparse
import base64
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import synth
from model import ImageEncoder, TextEncoder, IMG_SIZE


def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def q8(w, axes):
    axes = axes if isinstance(axes, tuple) else (axes,)
    keep = [i for i in range(w.ndim) if i not in axes]
    s = np.abs(w).max(axis=axes) / 127.0
    s = np.where(s == 0, 1e-8, s)
    shape = [1] * w.ndim
    for k, i in enumerate(keep):
        shape[i] = s.shape[k]
    return np.rint(w / s.reshape(shape)).clip(-127, 127).astype(np.int8), s.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="runs/tiny.pt")
    ap.add_argument("--out", default="docs/model.json")
    ap.add_argument("--root", default="data/cifar-10-batches-py")
    ap.add_argument("--demos", type=int, default=6)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, weights_only=False)
    if ck.get("data") in ("cifar", "stl", "trash"):
        D = __import__({"cifar": "data_real", "stl": "data_stl", "trash": "data_trash"}[ck["data"]])
    else:
        import synth as D
    up = getattr(D, "UPSCALE", 1)
    img = ImageEncoder(); img.load_state_dict(ck["img"]); img.eval()
    txt = TextEncoder(vocab=len(ck["vocab"])); txt.load_state_dict(ck["txt"]); txt.eval()

    convs = []
    for c in img.convs:
        q, s = q8(c.weight.detach().numpy(), (1, 2, 3))
        q = np.ascontiguousarray(q.transpose(0, 2, 3, 1))   # [oc][3][3][ic], the layout the JS reads
        convs.append({"w": b64(q, np.int8), "s": b64(s, np.float32),
                      "b": b64(c.bias.detach().numpy(), np.float32),
                      "ic": int(c.in_channels), "oc": int(c.out_channels)})
    fq, fs = q8(img.fc.weight.detach().numpy(), (1,))
    fc = {"w": b64(fq, np.int8), "s": b64(fs, np.float32),
          "b": b64(img.fc.bias.detach().numpy(), np.float32)}

    R, Rs = q8(ck["R"].numpy(), (1,))

    # a few frames + their true class, so the page works without an upload
    # demo gallery: one frame per class so a text query can actually find its match, and three of
    # them are frames the model gets wrong, so the page never pretends it is right every time
    n_cls = len(ck["names"])
    right, wrong = {}, {}
    imgs, txts, labels, names = D.make_dataset(600, seed=777, root=args.root)
    with torch.no_grad():
        for i in range(len(labels)):
            lab = int(labels[i])
            px = np.clip(imgs[0][i], 0, 255).astype(np.uint8)
            if up > 1:
                px = np.repeat(np.repeat(px, up, 0), up, 1)
            e = img(torch.from_numpy(px[None] / 1.0).permute(0, 3, 1, 2).float())[0]
            y = (e - ck["mu_a"]) @ ck["R"] + ck["mu_b"]
            p = F.normalize(y, dim=-1) @ ck["protos"].T
            hit = int(p.argmax()) == lab
            bucket = right if hit else wrong
            if lab in bucket:
                continue
            bucket[lab] = {"rgb": b64(px, np.uint8),
                           "true": ck["names"][lab],
                           "float_top": ck["names"][int(p.argmax())],
                           "emb": b64(e.numpy(), np.float32),
                           "rot": b64(y.numpy(), np.float32)}
            if len(right) >= n_cls and len(wrong) >= n_cls:
                break

    misses = [wrong[c] for c in sorted(wrong)][:3]
    demos = []
    for c in range(min(n_cls, args.demos)):
        src = right.get(c)
        if src is None:
            continue
        demos.append(src)
    # replace three of them with the same class's miss, so the gallery has real failures
    for j, miss in enumerate(misses):
        if j < len(demos):
            demos[j] = miss
    demos = demos[:args.demos]
    vocab = {w: i for w, i in ck["vocab"].items()}

    out = {
        "img": IMG_SIZE,
        "channels": [3] + [int(c.out_channels) for c in img.convs],
        "d_img": int(ck["R"].shape[0]),
        "d_txt": int(ck["R"].shape[1]),
        "convs": convs,
        "fc": fc,
        "txt": {
            "vocab": vocab,
            "emb": b64(txt.emb.weight.detach().numpy(), np.float32),
            "d_emb": int(txt.emb.embedding_dim),
            "fc1": {"w": b64(txt.fc1.weight.detach().numpy(), np.float32),
                    "b": b64(txt.fc1.bias.detach().numpy(), np.float32)},
            "fc2": {"w": b64(txt.fc2.weight.detach().numpy(), np.float32),
                    "b": b64(txt.fc2.bias.detach().numpy(), np.float32)},
        },
        "rot": {"R": b64(R, np.int8), "s": b64(Rs, np.float32),
                "mu_a": b64(ck["mu_a"].numpy(), np.float32),
                "mu_b": b64(ck["mu_b"].numpy(), np.float32)},
        "names": ck["names"],
        "protos": b64(ck["protos"].numpy(), np.float32),
        "metrics": {"cca": float(ck["cca"]), "acc": float(ck["acc"]),
                    "views": ck["views"], "mode": ck["mode"]},
        "demos": demos,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1024:.0f} KiB, {len(demos)} demo frames)")


if __name__ == "__main__":
    main()

"""Quantise the trained encoders to int8 and emit the C the ESP32-S3 compiles.

    python export_c.py runs/tiny.pt

Writes:
    firmware/prh_net.h / prh_net.c   int8 weights + per-channel requantisation constants
    firmware/prh_rot.h / prh_rot.c   the rotation R, mu_a, mu_b
    firmware/prh_proto.h/.c          int8 class prototypes and their names
    host/testvec.c                   one test image, its PyTorch embedding, its expected class

Everything the device runs is produced here; `prh_infer.c` (hand written) is the only moving part
that is not generated.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import synth
from model import ImageEncoder, TextEncoder, D_IMG, D_TXT, IMG_SIZE

MAXQ = 127.0


def qweight(w, axis):
    """Symmetric int8 weights with one scale per output channel (axis=0 or a tuple of axes)."""
    axes = axis if isinstance(axis, tuple) else (axis,)
    s = np.abs(w).max(axis=axes) / MAXQ
    s = np.where(s == 0, 1e-8, s)
    keep = [i for i in range(w.ndim) if i not in axes]
    shape = [1] * w.ndim
    for k, i in enumerate(keep):
        shape[i] = s.shape[k]
    q = np.rint(w / s.reshape(shape)).clip(-MAXQ, MAXQ).astype(np.int8)
    return q, s


def conv_int8(xq, wq, gain, biasq, stride=2, pad=1):
    """Exactly what prh_infer.c does: int8 x int8 -> int32 accumulate -> float requantise."""
    h, w, ic = xq.shape
    oc = wq.shape[0]
    xp = np.pad(xq.astype(np.int32), ((pad, pad), (pad, pad), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(0, 1))[::stride, ::stride]
    oh, ow = win.shape[0], win.shape[1]
    w2 = wq.transpose(1, 2, 3, 0)                      # [3,3,IC,OC]
    acc = np.tensordot(win, w2, axes=([2, 3, 4], [2, 0, 1]))   # [OH,OW,OC]
    y = np.rint(acc.astype(np.float32) * gain.reshape(1, 1, oc) + biasq.reshape(1, 1, oc))
    return np.clip(y, -MAXQ, MAXQ).astype(np.int8)


def linear_int8(xq, wq, gain, biasq):
    acc = xq.astype(np.int32) @ wq.astype(np.int32).T
    return acc


def calib_max(model, imgs, n=128, q=99.9):
    """Activation ranges for int8 scales, at the 99.9th percentile rather than the max: one bright
    outlier patch would otherwise crush the resolution of every other value in the tensor."""
    samples = {}
    hooks = []

    def mk(i):
        def hook(_m, _inp, out):
            v = out.detach().abs().flatten().cpu().numpy()
            if len(v) > 200_000:
                v = v[:: len(v) // 200_000 + 1]
            samples.setdefault(i, []).append(v)
        return hook

    for i, c in enumerate(model.convs):
        hooks.append(c.register_forward_hook(mk(i)))

    pooled = []
    h2 = model.fc.register_forward_pre_hook(
        lambda m, inp: pooled.append(inp[0].detach().abs().flatten().cpu().numpy()))

    with torch.no_grad():
        for i in range(0, min(n, len(imgs)), 16):
            model(torch.from_numpy(imgs[i:i + 16]).permute(0, 3, 1, 2).float())

    for h in hooks + [h2]:
        h.remove()

    acts = {i: float(np.percentile(np.concatenate(v), q)) for i, v in samples.items()}
    return acts, float(np.percentile(np.concatenate(pooled), q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="runs/tiny.pt")
    ap.add_argument("--outdir", default="firmware")
    ap.add_argument("--root", default="data/cifar-10-batches-py")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, weights_only=False)
    img = ImageEncoder()
    img.load_state_dict(ck["img"])
    img.eval()
    txt = TextEncoder(vocab=len(ck["vocab"]))
    txt.load_state_dict(ck["txt"])
    txt.eval()

    if ck.get("data") in ("cifar", "stl", "trash"):
        D = __import__({"cifar": "data_real", "stl": "data_stl", "trash": "data_trash"}[ck["data"]])
    else:
        import synth as D
    up = getattr(D, "UPSCALE", 1)
    imgs, txts, labels, names = D.make_dataset(400, seed=args.seed + 12345, root=args.root)
    if up > 1:
        imgs = np.repeat(np.repeat(imgs, up, axis=2), up, axis=3)
    acts, pool_max = calib_max(img, imgs[0])

    scales = [max(acts[i], 1e-4) / MAXQ for i in range(len(img.convs))]
    s_in = 1.0 / MAXQ                      # input: pixel/128 - 1 in [-1, 0.992] quantised to int8
    s_pool = max(pool_max, 1e-6) / MAXQ

    # ── quantise ──
    layers = []
    for i, c in enumerate(img.convs):
        w = c.weight.detach().cpu().numpy()                 # [OC,IC,3,3]
        b = c.bias.detach().cpu().numpy()
        q, sw = qweight(w, axis=(1, 2, 3))
        gain = sw * s_in / scales[i]
        biasq = b / scales[i]
        layers.append({
            "q": q.transpose(0, 2, 3, 1).copy(),             # [OC,3,3,IC]
            "gain": gain.astype(np.float32), "biasq": biasq.astype(np.float32),
            "oc": w.shape[0], "ic": w.shape[1],
        })
        s_in = scales[i]

    fw = img.fc.weight.detach().cpu().numpy()
    fb = img.fc.bias.detach().cpu().numpy()
    fq, fsw = qweight(fw, axis=1)
    fc_gain = (fsw * s_pool / 1.0).astype(np.float32)        # output stays float
    fc_bias = fb.astype(np.float32)

    R = ck["R"].numpy().astype(np.float32)
    mu_a, mu_b = ck["mu_a"].numpy(), ck["mu_b"].numpy()
    Rq, Rs = qweight(R, axis=1)

    protos = ck["protos"].numpy().astype(np.float32)
    pn = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    ps = np.abs(pn).max() / MAXQ
    pq = np.rint(pn / ps).clip(-MAXQ, MAXQ).astype(np.int8)

    # ── simulate the int8 path and compare with float ──
    def int8_embed(rgb):
        x = np.rint((rgb / 128.0 - 1.0) * MAXQ).clip(-MAXQ, MAXQ).astype(np.int8)
        for L in layers:
            x = conv_int8(x, L["q"], L["gain"], L["biasq"])
            np.maximum(x, 0, out=x)                          # ReLU
        pooled = x.astype(np.float32).mean(axis=(0, 1)) * scales[-1]
        xq = np.rint(pooled / s_pool).clip(-MAXQ, MAXQ).astype(np.int8)
        acc = linear_int8(xq, fq, fc_gain, fc_bias)
        return acc.astype(np.float32) * fc_gain + fc_bias

    test = imgs[0][:64]
    with torch.no_grad():
        ref = img(torch.from_numpy(test).permute(0, 3, 1, 2).float()).numpy()
    got = np.stack([int8_embed(t) for t in test])
    cos = np.sum(ref * got, 1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1) + 1e-9)

    print(f"int8 vs float encoder embedding: mean cosine {cos.mean():.4f}  (min {cos.min():.4f})")

    final_hw = (IMG_SIZE // 2 ** len(layers)) ** 2
    max_act = max((IMG_SIZE // 2 ** (i + 1)) ** 2 * layers[i]["oc"] for i in range(len(layers)))
    os.makedirs(args.outdir, exist_ok=True)
    _write_net(args.outdir, layers, fc_gain, fc_bias, fq, scales, s_pool, final_hw, max_act)
    _write_rot(args.outdir, Rq, Rs, mu_a, mu_b, R)
    _write_proto(args.outdir, pq, ps, ck["names"])
    _write_testvec(args.outdir, test[0], ref[0], got[0], ck, labels, imgs, txts, img, txt)

    total = sum(L["q"].size for L in layers) + fq.size + Rq.size + pq.size
    print(f"float32 weights: {sum(p.numel() for p in img.parameters()) * 4 / 1024:.0f} KiB"
          f"   ->  int8 on flash: {total / 1024:.0f} KiB (+ scales)")
    print("wrote firmware/prh_net.[ch], prh_rot.[ch], prh_proto.[ch], host/testvec.c")


def _arr(name, a, ctype="int8_t"):
    flat = a.reshape(-1)
    body = ",".join(str(int(v)) for v in flat) if ctype == "int8_t" else \
        ",".join(f"{float(v):.8g}f" for v in flat)
    return f"const {ctype} {name}[{flat.size}] = {{{body}}};\n"


def _write_net(outdir, layers, fc_gain, fc_bias, fq, scales, s_pool, final_hw, max_act):
    hdr = [f"""/* GENERATED by export_c.py — do not edit.
 * Image encoder {IMG_SIZE}x{IMG_SIZE}x3 -> {D_IMG}, int8 weights, int32 accumulation, NHWC.
 */
#pragma once
#include <stdint.h>

#define PRH_IMG      {IMG_SIZE}
#define PRH_NCONV    {len(layers)}
#define PRH_D        {D_IMG}
#define PRH_CH_LAST  {layers[-1]["oc"]}
#define PRH_FINAL_HW {final_hw}
#define PRH_MAXACT   {max_act}

typedef struct {{
    const int8_t *w;      /* [oc][3][3][ic] */
    const float  *gain;   /* per output channel */
    const float  *bias;
    int ic, oc;
}} prh_conv_t;

extern const prh_conv_t PRH_CONVS[PRH_NCONV];
extern const float PRH_SCALES[PRH_NCONV];
extern const float PRH_S_POOL;
extern const int8_t PRH_FC[PRH_D * PRH_CH_LAST];
extern const float  PRH_FC_G[PRH_D];
extern const float  PRH_FC_B[PRH_D];
"""]
    src = ['#include "prh_net.h"\n\n']
    names = []
    for i, L in enumerate(layers):
        src.append(_arr(f"W{i}", L["q"]))
        src.append(_arr(f"G{i}", L["gain"], "float"))
        src.append(_arr(f"B{i}", L["biasq"], "float"))
        names.append(i)
    src.append("const prh_conv_t PRH_CONVS[PRH_NCONV] = {\n" + "".join(
        f"    {{ W{i}, G{i}, B{i}, {layers[i]['ic']}, {layers[i]['oc']} }},\n" for i in names) + "};\n")
    src.append(_arr("PRH_SCALES", np.array(scales, np.float32), "float"))
    src.append(f"const float PRH_S_POOL = {s_pool:.8g}f;\n")
    src.append(_arr("PRH_FC", fq))
    src.append(_arr("PRH_FC_G", fc_gain, "float"))
    src.append(_arr("PRH_FC_B", fc_bias, "float"))
    open(os.path.join(outdir, "prh_net.h"), "w").write("".join(hdr))
    open(os.path.join(outdir, "prh_net.c"), "w").write("".join(src).replace("const int8_t W", "static const int8_t W")
        .replace("const float G", "static const float G").replace("const float B", "static const float B"))


def _write_rot(outdir, Rq, Rs, mu_a, mu_b, R):
    da, db = Rq.shape
    hdr = f"""/* GENERATED by export_c.py — do not edit.
 * The one rotation R the paper leaves behind: x -> (x - mu_a) R + mu_b.
 */
#pragma once
#include <stdint.h>

#define PRH_DA {da}
#define PRH_DB {db}
#define PRH_X_SCALE 128.0f

extern const int8_t PRH_R[{Rq.size}];
extern const float  PRH_R_SCALE[{da}];
extern const float  PRH_MU_A[{da}];
extern const float  PRH_MU_B[{db}];
"""
    ser = [f'#include "prh_rot.h"\n\n', _arr("PRH_R", Rq), _arr("PRH_R_SCALE", Rs, "float"),
           _arr("PRH_MU_A", mu_a, "float"), _arr("PRH_MU_B", mu_b, "float")]
    open(os.path.join(outdir, "prh_rot.h"), "w").write(hdr)
    open(os.path.join(outdir, "prh_rot.c"), "w").write("".join(ser))


def _write_proto(outdir, pq, ps, names):
    n, d = pq.shape
    hdr = f"""/* GENERATED by export_c.py — do not edit. */
#pragma once
#include <stdint.h>
#define PRH_NCLASS {n}
#define PRH_PSCALE {ps:.8g}f
extern const int8_t PRH_PROTO[{pq.size}];
extern const char *PRH_NAMES[{n}];
"""
    src = ['#include "prh_proto.h"\n\n', _arr("PRH_PROTO", pq)]
    src.append("const char *PRH_NAMES[PRH_NCLASS] = {" + ",".join(f'"{s}"' for s in names) + "};\n")
    open(os.path.join(outdir, "prh_proto.h"), "w").write(hdr)
    open(os.path.join(outdir, "prh_proto.c"), "w").write("".join(src))


def _write_testvec(outdir, img0, ref0, got0, ck, labels, imgs, txts, img, txt):
    host = os.path.join(os.path.dirname(outdir), "host")
    os.makedirs(host, exist_ok=True)
    px = img0.astype(np.uint8).reshape(-1)

    with torch.no_grad():
        e = img(torch.from_numpy(imgs[0][:1]).permute(0, 3, 1, 2).float())[0].numpy()
    y = (e - ck["mu_a"].numpy()) @ ck["R"].numpy() + ck["mu_b"].numpy()
    p = F.normalize(torch.from_numpy(y), dim=-1) @ ck["protos"].T
    cls = int(p.argmax())

    src = f"""/* GENERATED by export_c.py — one test vector for host/test_infer.c */
#include <stdint.h>
const uint8_t PRH_TEST_IMG[{px.size}] = {{{",".join(str(int(v)) for v in px)}}};
const float PRH_TEST_EMB[{len(e)}] = {{{",".join(f"{float(v):.6g}f" for v in e)}}};
const char *PRH_TEST_CLASS = "{ck['names'][cls]}";
"""
    open(os.path.join(host, "testvec.c"), "w").write(src)


if __name__ == "__main__":
    main()

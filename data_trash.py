"""Real waste photographs: TrashNet, six classes, 96x96, with synonym captions.

    kaggle datasets download -d feyzazkefe/trashnet -p data/trash --unzip

The point of this dataset for the project: the classes are things you can hold in front of an
ESP32-CAM (a bottle, a can, a sheet of paper), and the six names are ordinary words, so the whole
"a class is a sentence" story survives contact with a real object.
"""

import os
import pickle

import numpy as np
from PIL import Image

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
UPSCALE = 1
SIZE = 96

SYN = {
    "cardboard": ["cardboard", "a cardboard box", "a box", "packaging"],
    "glass": ["glass", "a glass bottle", "a jar", "glassware"],
    "metal": ["metal", "a metal can", "a tin can", "a can"],
    "paper": ["paper", "a sheet of paper", "a newspaper", "paper waste"],
    "plastic": ["plastic", "a plastic bottle", "a plastic cup", "plastic waste"],
    "trash": ["trash", "garbage", "rubbish", "waste"],
}
FILLER = ["a", "photo", "of", "the", "in", "picture", "this", "shows", "is", "an"]


def build_vocab():
    words = sorted({w for v in SYN.values() for s in v for w in s.split()} | set(FILLER) | set(CLASSES))
    return {w: i for i, w in enumerate(words)}


VOCAB = build_vocab()
PAD = len(VOCAB)


def _load_all(root, cache="/tmp/trashnet96.pkl"):
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    xs, ys = [], []
    for ci, name in enumerate(CLASSES):
        d = os.path.join(root, name)
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            im = Image.open(os.path.join(d, fn)).convert("RGB")
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            im = im.resize((SIZE, SIZE), Image.BILINEAR)
            xs.append(np.asarray(im, np.uint8))
            ys.append(ci)
    out = (np.stack(xs), np.array(ys, np.int64))
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out


def augment(rng, img, train=True):
    x = img.astype(np.float32)
    if not train:
        return img
    if rng.random() < 0.5:
        x = x[:, ::-1]
    c = 84
    y0, x0 = rng.integers(0, SIZE - c + 1, size=2)
    x = x[y0:y0 + c, x0:x0 + c]
    idx = (np.arange(SIZE) * c / SIZE).astype(int)
    x = x[idx][:, idx]
    x *= rng.uniform(0.8, 1.25)
    x += rng.normal(0, 5, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def caption(rng, label, max_len=8):
    ids = []
    if rng.random() < 0.5:
        ids.append(VOCAB[rng.choice(FILLER[:2])])
    phrase = rng.choice(SYN[CLASSES[label]]).split()
    ids += [VOCAB[w] for w in phrase if w in VOCAB]
    if rng.random() < 0.3:
        ids.append(VOCAB[rng.choice(FILLER[5:])])
    ids = ids[:max_len]
    return ids + [PAD] * (max_len - len(ids))


def make_dataset(n, seed=0, max_len=8, root="data/trash/dataset-resized"):
    rng = np.random.default_rng(seed)
    x, y = _load_all(root)
    idx = rng.permutation(len(x))[:n]
    imgs = np.stack([[augment(rng, x[i]) for i in idx],
                     [augment(rng, x[i]) for i in idx]])
    txts = np.stack([[caption(rng, int(y[i]), max_len) for i in idx],
                     [caption(rng, int(y[i]), max_len) for i in idx]])
    return imgs, txts, y[idx], CLASSES


def test_set(root="data/trash/dataset-resized", n=400, seed=0, max_len=8):
    rng = np.random.default_rng(seed + 991)
    x, y = _load_all(root)
    idx = rng.permutation(len(x))[:n]
    return x[idx], y[idx]


def class_prototype_sentences(rng, per_class=8, max_len=8):
    out = []
    for c in range(len(CLASSES)):
        sents = []
        for _ in range(per_class):
            ids = [VOCAB[w] for w in rng.choice(SYN[CLASSES[c]]).split() if w in VOCAB]
            sents.append(ids + [PAD] * (max_len - len(ids)))
        out.append(sents)
    return out

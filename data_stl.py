"""Real 96x96 photographs: STL-10's labelled split, with synonym captions.

STL-10 is the right size for this model: 96x96 is the camera's native input, so the encoder sees
real pixels instead of a 3x upscale, and the ten classes are photographic scenes rather than
renders. Files come from `kaggle datasets download -d yellowflag/stl10labeled2`.

Layout expected in --root:
    train_X.bin  train_y.bin  test_X.bin  test_y.bin  class_names.txt
"""

import os

import numpy as np

UPSCALE = 1

SYN = {
    "airplane": ["airplane", "plane", "aircraft", "jet"],
    "bird": ["bird", "songbird", "sparrow"],
    "car": ["car", "automobile", "vehicle", "sedan"],
    "cat": ["cat", "kitten", "kitty"],
    "deer": ["deer", "stag", "doe"],
    "dog": ["dog", "puppy", "hound"],
    "horse": ["horse", "pony", "stallion"],
    "monkey": ["monkey", "ape", "primate"],
    "ship": ["ship", "boat", "vessel", "liner"],
    "truck": ["truck", "lorry", "van"],
}
FILLER = ["a", "photo", "of", "the", "in", "picture", "this", "shows", "is", "an"]


def _vocab(class_names):
    words = sorted({w for v in SYN.values() for w in v} | set(FILLER) |
                   {c for c in class_names})
    return {w: i for i, w in enumerate(words)}


def _read(root, split):
    x = np.fromfile(os.path.join(root, f"{split}_X.bin"), dtype=np.uint8)
    y = np.fromfile(os.path.join(root, f"{split}_y.bin"), dtype=np.uint8)
    n = len(y)
    return x.reshape(n, 3, 96, 96).transpose(0, 2, 3, 1), y.astype(np.int64) - 1


def load(root):
    with open(os.path.join(root, "class_names.txt")) as f:
        names = [w.strip() for w in f if w.strip()]
    xtr, ytr = _read(root, "train")
    xte, yte = _read(root, "test")
    return xtr, ytr, xte, yte, names


def augment(rng, img, train=True):
    x = img.astype(np.float32)
    if not train:
        return img
    if rng.random() < 0.5:
        x = x[:, ::-1]
    c = 84
    y0, x0 = rng.integers(0, 96 - c + 1, size=2)
    x = x[y0:y0 + c, x0:x0 + c]
    idx = (np.arange(96) * c / 96).astype(int)
    x = x[idx][:, idx]
    x *= rng.uniform(0.8, 1.25)
    x += rng.normal(0, 6, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def make_dataset(n, seed=0, max_len=8, root="data/stl"):
    rng = np.random.default_rng(seed)
    xtr, ytr, _, _, names = load(root)
    idx = rng.permutation(len(xtr))[:n]
    vocab = _vocab(names)

    def caption(label):
        ids = []
        if rng.random() < 0.6:
            ids.append(vocab[rng.choice(FILLER[:2])])
        if rng.random() < 0.5:
            ids.append(vocab[rng.choice(FILLER[2:5])])
        ids.append(vocab[rng.choice(SYN[names[label]])])
        if rng.random() < 0.3:
            ids.append(vocab[rng.choice(FILLER[5:])])
        ids = ids[:max_len]
        return ids + [len(vocab)] * (max_len - len(ids))

    imgs = np.stack([[augment(rng, xtr[i]) for i in idx],
                     [augment(rng, xtr[i]) for i in idx]])
    txts = np.stack([[caption(int(ytr[i])) for i in idx],
                     [caption(int(ytr[i])) for i in idx]])
    VOCAB.clear()
    VOCAB.update(vocab)
    return imgs, txts, ytr[idx], names


VOCAB = {}
PAD = 0


def test_set(root="data/stl", n=2000, seed=0, max_len=8):
    rng = np.random.default_rng(seed + 991)
    _, _, xte, yte, _ = load(root)
    idx = rng.permutation(len(xte))[:n]
    return xte[idx], yte[idx]


def class_prototype_sentences(rng, per_class=8, max_len=8):
    n = len(VOCAB)
    out = []
    for c in range(len(SYN)):
        name = list(SYN)[c]
        sents = []
        for _ in range(per_class):
            ids = [VOCAB[rng.choice(SYN[name])]]
            sents.append(ids + [n] * (max_len - len(ids)))
        out.append(sents)
    return out

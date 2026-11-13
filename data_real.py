"""Real photographs: CIFAR-10 at 96x96 (3x nearest upscale), with synonym captions.

The image side sees two augmented crops of the same photograph — the paper's multi-crop image arm.
The text side sees two different sentences naming the same class — genuine good views. That is the
same asymmetry the paper has, and it is what the demo needs: the classes are real objects, and a
class is still just a sentence.

Download once:  curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz && tar xzf it
"""

import os
import pickle

import numpy as np

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

SYN = {
    "airplane": ["airplane", "plane", "aircraft", "jet"],
    "automobile": ["automobile", "car", "vehicle", "sedan"],
    "bird": ["bird", "songbird", "sparrow"],
    "cat": ["cat", "kitten", "kitty"],
    "deer": ["deer", "stag", "doe"],
    "dog": ["dog", "puppy", "hound"],
    "frog": ["frog", "toad", "amphibian"],
    "horse": ["horse", "pony", "stallion"],
    "ship": ["ship", "boat", "vessel", "liner"],
    "truck": ["truck", "lorry", "van"],
}
FILLER = ["a", "photo", "of", "the", "in", "picture", "this", "shows", "is", "an"]

UPSCALE = 3          # 32 -> 96, done on the train side so the cache stays small


def build_vocab():
    words = sorted({w for vs in SYN.values() for w in vs} | set(FILLER))
    return {w: i for i, w in enumerate(words)}


VOCAB = build_vocab()
PAD = len(VOCAB)


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar(root):
    """(train_x, train_y, test_x, test_y), uint8 NHWC 32x32x3."""
    xs, ys = [], []
    for i in range(1, 6):
        d = _unpickle(os.path.join(root, f"data_batch_{i}"))
        xs.append(d[b"data"]); ys.append(d[b"labels"])
    x = np.concatenate(xs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.concatenate(ys)
    t = _unpickle(os.path.join(root, "test_batch"))
    xt = t[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    yt = np.array(t[b"labels"])
    return x, y, xt, yt


def caption(rng, label, max_len=8):
    ids = []
    if rng.random() < 0.6:
        ids.append(VOCAB[rng.choice(FILLER[:2])])
    if rng.random() < 0.5:
        ids.append(VOCAB[rng.choice(FILLER[2:5])])
    ids.append(VOCAB[rng.choice(SYN[CLASSES[label]])])
    if rng.random() < 0.3:
        ids.append(VOCAB[rng.choice(FILLER[5:])])
    ids = ids[:max_len]
    return ids + [PAD] * (max_len - len(ids))


def augment(rng, img, train=True):
    """A 32x32 photograph -> one view: crop, flip, brightness, noise."""
    x = img.astype(np.float32)
    if train:
        if rng.random() < 0.5:
            x = x[:, ::-1]
        y0, x0 = rng.integers(0, 5, size=2)
        x = x[y0:y0 + 28, x0:x0 + 28]
        idx = (np.arange(32) * 28 / 32).astype(int)      # nearest resize back to 32x32
        x = x[idx][:, idx]
        x *= rng.uniform(0.75, 1.3)
        x += rng.normal(0, 6, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def make_dataset(n, seed=0, max_len=8, root="cifar-10-batches-py"):
    """Two views per photograph, two sentences per class label."""
    rng = np.random.default_rng(seed)
    x, y, _, _ = load_cifar(root)
    idx = rng.permutation(len(x))[:n]
    imgs = np.stack([[augment(rng, x[i]) for i in idx],
                     [augment(rng, x[i]) for i in idx]])
    txts = np.stack([[caption(rng, int(y[i]), max_len) for i in idx],
                     [caption(rng, int(y[i]), max_len) for i in idx]])
    return imgs, txts, y[idx].astype(np.int64), CLASSES


def test_set(root="cifar-10-batches-py", n=2000, seed=0, max_len=8):
    rng = np.random.default_rng(seed + 991)
    _, _, xt, yt = load_cifar(root)
    idx = rng.permutation(len(xt))[:n]
    imgs = np.stack([np.repeat(np.repeat(xt[i].astype(np.float32), UPSCALE, 0), UPSCALE, 1)
                     for i in idx])
    return imgs.astype(np.uint8), yt[idx].astype(np.int64)


def class_prototype_sentences(rng, per_class=8, max_len=8):
    return [[caption(rng, c, max_len) for _ in range(per_class)] for c in range(len(CLASSES))]

"""Synthetic two-modality world: the same world state rendered as an image and as a caption.

The `prh-proof` paper needs two modalities that are both functions of one world state, and whose
positive pairs are two independent observations of that state. A procedural scene gives exactly
that, with no dataset download: the world state is (shape, colour, size, position, rotation), the
image modality renders it, the text modality names it with a random synonym per attribute.

Swap `SceneDataset` for a real loader (Flickr8k/COCO images + 5 captions) when you have one; the
rest of the pipeline does not change.
"""

import numpy as np

SHAPES = ["circle", "square", "triangle", "cross"]
COLORS = ["red", "green", "blue", "yellow", "purple", "orange"]
SIZES = ["small", "big"]

SYN = {
    "circle": ["circle", "disk", "ring", "round shape"],
    "square": ["square", "box", "block", "rectangle"],
    "triangle": ["triangle", "wedge", "pyramid"],
    "cross": ["cross", "plus", "x shape"],
    "red": ["red", "crimson", "scarlet"],
    "green": ["green", "emerald", "lime"],
    "blue": ["blue", "azure", "navy"],
    "yellow": ["yellow", "gold", "amber"],
    "purple": ["purple", "violet", "magenta"],
    "orange": ["orange", "coral", "tangerine"],
    "small": ["small", "tiny", "little"],
    "big": ["big", "large", "huge"],
}
FILLER = ["a", "the", "photo", "of", "with", "on", "side", "view"]

RGB = {
    "red": (220, 40, 40), "green": (40, 190, 70), "blue": (50, 90, 220),
    "yellow": (225, 205, 50), "purple": (150, 70, 200), "orange": (235, 140, 40),
}


def build_vocab():
    words = sorted({w for vs in SYN.values() for w in vs} | set(FILLER))
    return {w: i for i, w in enumerate(words)}


VOCAB = build_vocab()
PAD = len(VOCAB)


def caption(rng, state, max_len=8):
    """One observation of the world state in the text modality: a sentence, not a fixed template."""
    ids = []
    if rng.random() < 0.5:
        ids.append(VOCAB[rng.choice(FILLER[:2])])
    ids.append(VOCAB[rng.choice(SYN[SIZES[state["size"]]])])
    ids.append(VOCAB[rng.choice(SYN[COLORS[state["color"]]])])
    ids.append(VOCAB[rng.choice(SYN[SHAPES[state["shape"]]])])
    if rng.random() < 0.35:
        ids.append(VOCAB[rng.choice(FILLER[2:])])
    ids = ids[:max_len]
    return ids + [PAD] * (max_len - len(ids))


def render(rng, state, size=96):
    """One observation of the world state in the image modality."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = size / 2 + state["px"] * size * 0.28
    cy = size / 2 + state["py"] * size * 0.28
    rad = (12.0 if SIZES[state["size"]] == "small" else 21.0) * state["scale"]

    dx, dy = xx - cx, yy - cy
    th = state["rot"]
    rx, ry = dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)

    if state["shape"] == 0:
        mask = rx * rx + ry * ry <= rad * rad
    elif state["shape"] == 1:
        mask = (np.abs(rx) <= rad) & (np.abs(ry) <= rad)
    elif state["shape"] == 2:
        band = (ry <= rad) & (ry >= -rad)
        half = (ry + rad) * 0.5
        mask = band & (np.abs(rx) <= half)
    else:
        mask = ((np.abs(rx) <= rad) & (np.abs(ry) <= rad / 3)) | \
               ((np.abs(ry) <= rad) & (np.abs(rx) <= rad / 3))

    img = np.zeros((size, size, 3), np.float32) + 18.0

    # a nuisance patch: present in one view, often absent in the other, never named in a caption
    px, py = rng.integers(0, size - 20), rng.integers(0, size - 20)
    pw, ph = rng.integers(8, 26), rng.integers(8, 26)
    img[py:py + ph, px:px + pw] = np.array(RGB[COLORS[int(rng.integers(0, 6))]], np.float32) * 0.45

    base = np.array(RGB[COLORS[state["color"]]], np.float32) * state["bright"]
    img[mask] = base

    img += rng.normal(0, 12, img.shape).astype(np.float32)
    img *= rng.uniform(0.6, 1.4)
    return np.clip(img, 0, 255)


def sample_state(rng):
    return {
        "shape": int(rng.integers(0, len(SHAPES))),
        "color": int(rng.integers(0, len(COLORS))),
        "size": int(rng.integers(0, len(SIZES))),
        "px": float(rng.uniform(-1, 1)), "py": float(rng.uniform(-1, 1)),
        "rot": float(rng.uniform(-np.pi, np.pi)),
        "scale": float(rng.uniform(0.8, 1.2)),
        "bright": float(rng.uniform(0.45, 1.5)),
    }


def make_dataset(n, seed=0, max_len=8):
    """[n] world states, each with two images and two captions (two independent observations)."""
    rng = np.random.default_rng(seed)
    states = [sample_state(rng) for _ in range(n)]
    imgs = np.stack([[render(rng, s) for s in states], [render(rng, s) for s in states]])
    txts = np.stack([[caption(rng, s, max_len) for s in states],
                     [caption(rng, s, max_len) for s in states]])
    labels = np.array([s["shape"] * len(COLORS) + s["color"] for s in states])
    return imgs, txts, labels, states


def class_prototype_sentences(rng, per_class=8, max_len=8):
    """For zero-shot: sentences naming each class, no images involved."""
    out = []
    for shape in range(len(SHAPES)):
        for color in range(len(COLORS)):
            sents = []
            for _ in range(per_class):
                st = {"shape": shape, "color": color, "size": int(rng.integers(0, 2))}
                sents.append(caption(rng, st, max_len))
            out.append(sents)
    return out

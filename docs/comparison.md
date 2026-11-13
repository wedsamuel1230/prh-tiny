# The three arms, and what the gap means

TrashNet, six classes, 96×96, one architecture (42,960 parameters), 900 optimisation steps.

| arm | accuracy | image labels | wall clock |
|---|---|---|---|
| supervised, end to end | 58.1 % | 1,895 | 3.6 min |
| paper's method, classes as sentences | 46.9 % | 0 | 4.0 min |
| paper's method + 20-shot kNN | 49.2 % | 120 | 0.1 min |
| frozen encoder, linear probe | 55.7 % | 1,895 | 10 s |

Train and test are disjoint photographs (1,895 / 632). An earlier version of this file sampled the
test set from the same pool as training and reported 66.3 / 48.0 / 48.8 — those numbers were
inflated by leakage and are withdrawn.

Chance is 16.7 %. Reproduce with `python compare.py --data trash --root ... --ckpt runs/trash.pt`.

## Reading it

**Supervised wins by 11 points.** Nobody should be surprised: it sees every label, and the task is
closed-set. If your classes are fixed and labelled, plain cross-entropy is the right tool and this
repository has nothing to offer you.

**The paper's method reaches 2.8× chance with zero image labels.** The six class names — *cardboard,
glass, metal, paper, plastic, trash* — are the entire specification. The encoder learned from
photographs; the text encoder learned from the words; one rotation joined them, fitted on 512 pairs.

**Twenty labelled photos per class buy 2.3 points.** That is the interesting number for an embedded
project. Once the two spaces are aligned, adding labels is cheap but nearly pointless at this scale —
the bottleneck is the *alignment*, not the labels. If you need more accuracy, spend the effort on the
vision encoder and the training signal, not on more photographs.

## Why the label-free arm still loses

1. **Rank is low.** A 43 k CNN over five stride/conv layers produces a 64-dimensional embedding of a
   scene that genuinely needs more capacity. The rotation cannot recover information the image
   encoder threw away.
2. **The positive pairs are weak on the image side.** Two crops of one photograph (the paper's
   multi-crop arm) teach invariance to crop and brightness, not "two photographs of the same object".
   With one photo per item, that is the only pair available.
3. **The text side is 43 words.** Six classes with four synonyms each is a thin description of the
   world. On the paper's 940 M web images the text side has seen everything; here it has seen
   *bottle*, *jar*, *can*.

## What would move the numbers

- A wider image encoder (12–20 k → 200 k parameters) with the same export path. The flash budget is
  13 MiB; the model uses 59 KiB. Capacity is free; accuracy is not.
- More steps and a real GPU. 3.6 minutes on a laptop CPU is a probe, not a training run.
- Better positives on the image side: two photographs of one object instead of two crops of one.

The export pipeline (`export_c.py`, `export_web.py`, `make test`) does not change for any of those.

# What is being tested, and how faithfully

## The paper in one paragraph

`Proving and Improving the Platonic Representation Hypothesis with JEPAs` (Balestriero & LeCun,
arXiv 2610.39302) proves that two encoders trained **separately**, one per modality, with a JEPA
objective, learn the *same* embedding up to one rotation — provided their positive pairs are two
observations of the same world state (*good views*). If the positives are instead corruptions of a
single observation (*bad views*, e.g. masked spans, i.e. next-token prediction), alignment is
bounded by how much the two modalities' positive-pair kernels differ.

Their table of consequences:

| claim | how this repo checks it |
|---|---|
| separate training suffices — no cross-modal pairs needed | two encoders, two optimisers, no shared batch |
| alignment is one orthogonal map `R` | Procrustes on 512 held-out states, residual reported |
| good views align, bad views do not | `--views good` vs `--views bad`, same seed |
| the map is the only thing you need downstream | 24-class zero-shot kNN through `R`, no image label ever used |

## Mapping

| paper | here | why |
|---|---|---|
| DINOv2 / DINOv3 / 109M text JEPA | 55k image CNN, 44k text encoder | must fit 16 MB flash and 7 ms of compute |
| alt-texts of 940 M images | procedural scenes + synonym captions | no download, runs in minutes, deterministic |
| LeJEPA (invariance + SIGReg) | invariance on RMS-normalised embeddings + SIGReg | same objective, numerically stable at this scale |
| one rotation fitted on 30k pairs | one rotation fitted on 512 pairs | the paper's point: pairs are only needed for `R` |
| LLM decoder for captions | 24 text prototypes, cosine kNN | an LLM cannot run on an S3 |

## The three deliberate deviations

1. **Scale.** 55k against 109M parameters, 2.5k training states against 940M images. This tests the
   *mechanism*, not the paper's numbers. Nothing here should be quoted as a reproduction.
2. **The data.** The default run uses CIFAR-10: 32×32 photographs upscaled to 96×96, ten real
   classes, and captions drawn from synonym sets ("a photo of a puppy", "the hound is in the
   picture"). The image side gets two augmented crops of one photograph — the paper's multi-crop
   image arm; the text side gets two different sentences for the same class — genuine good views.
   `--data synth` swaps in a procedural renderer that needs no download. Either way, replacing
   `make_dataset` with Flickr8k or COCO changes nothing downstream.
3. **No predictor, no projector.** The JEPA loss is applied to the encoder output directly. That
   keeps the C identical to the PyTorch module, which is the whole point of the int8 export.

## Quantisation

Weights: symmetric int8, one scale per output channel. Activations: symmetric int8, one scale per
tensor, calibrated on 400 real scenes. Convolution accumulates in int32 and requantises with one
`acc * gain + bias` per output element, ReLU fused into the clamp. `host/test_infer.c` compares the
C result against PyTorch on a held-out frame and fails the build if the cosine drops below 0.99 or
the predicted class changes.

## What a negative or weak result would mean

If alignment is high but zero-shot accuracy is low, the shared space is real but the text prototypes
are a poor readout — that is a *decoder* problem, not a PRH problem. If alignment is low, the two
positive-pair kernels genuinely differ, which at this scale usually means the image encoder is
capturing a nuisance (brightness, background) that the captions never mention. The two are
distinguishable with the numbers `train.py` prints, which is why it prints both.

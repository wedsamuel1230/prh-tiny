# prh-tiny — the Platonic Representation Hypothesis, on an ESP32-S3

An EdgeAI / TinyML experiment: take a paper about representations ([Balestriero & LeCun,
arXiv 2610.39302](https://arxiv.org/abs/2610.39302)), shrink it by four orders of magnitude, and see
what survives on a $5 microcontroller with a camera.

What survives is not the model — it is the **rotation**. Two 43,000-parameter encoders are trained
separately, on different modalities, with no paired data. One orthogonal map (`R`, 16 KiB) joins
their spaces, so a class is a sentence rather than a labelled dataset. The result ships as **59 KiB
of int8 weights**, runs in **471 ms** per 96×96 frame on an ESP32-S3-N16R8, and can be re-aimed at a
new class by editing text.

```
camera ──► int8 CNN (43 k) ──► 64-d ──► R (64→256) ──► cosine kNN ──► serial
```

## Does it beat plain machine learning?

No — and the repository exists to say so with numbers instead of adjectives. Six real waste
categories (TrashNet), one architecture, one step budget, three arms:

| arm | accuracy | image labels used | how long |
|---|---|---|---|
| plain supervised, same CNN end to end | **66.3 %** | all 2,500 | 2.3 min |
| the paper's method, classes as sentences | **48.0 %** | **0** | 3.6 min |
| the paper's method + 20 labelled photos per class | **48.8 %** | 120 | 0.1 min |

Chance is 16.7 %. Same encoder (42,960 parameters), same 900 optimisation steps, same photographs.

Read the table as the project's conclusion:

- **The method costs accuracy.** A supervised head on the same CNN, trained on the same photos for
  the same number of steps, is better. The gap is the price of never using an image–label pair.
- **The method buys editability.** The class list is text. Nothing about "plastic" is baked into the
  weights; the prototypes are sentences, so a class can be renamed, removed, or added without
  touching the model or the flash image.
- **The cost is at deployment-zero.** Once trained, the device carries the same 59 KiB either way,
  and the rotation is the cheapest object in the pipeline (16 KiB, 0.3 % of a forward pass).

If your classes are fixed and your labels exist, use supervised learning. This repository is for the
case where they are not, and for the question of whether the theory survives at 43 k parameters.

## What is tested, and how faithfully

| the paper | here |
|---|---|
| DINOv2 / DINOv3 + 109 M text JEPA | 43 k image CNN + 44 k text encoder (offline only) |
| 940 M web images, 30 k pairs for `R` | 2,500 photographs, 512 pairs for `R` |
| LeJEPA: invariance + SIGReg | invariance + SIGReg on RMS-normalised embeddings |
| one rotation, then an LLM decoder | one rotation, then cosine against text prototypes |
| GPU cluster | one laptop CPU, 3.6 minutes |

It tests the *mechanism*: separate training, one rotation, classes as sentences. It reproduces none
of the paper's numbers, and it never will at this size. [`docs/design.md`](docs/design.md) lists the
deliberate deviations.

## Hardware, measured

| item | value |
|---|---|
| board | ESP32-S3-N16R8, OV2640 (ESP32-S3-CAM pin map) |
| flash used by the model | **59 KiB** (43 KiB encoder + 16 KiB rotation + 2.5 KiB prototypes) |
| activation buffers | 108 KiB of int8 in SRAM, no encoder traffic to PSRAM |
| inference, 96×96 RGB565 | **471 ms** @ 240 MHz (naive C; esp-dl-class kernels would reach ~10 ms) |
| camera | `Detected OV2640 camera`, 8 MB octal PSRAM, 80 MHz |

The 471 ms is the honest figure for [the C in this repo](firmware/prh_infer.c): 5.6 M MAC at
~12 MMAC/s. esp-dl's hand-written int8 kernels reach ~0.77 GMAC/s on the same chip, so the same
model with a SIMD kernel is a ~10 ms problem. The kernel — not the model — is the remaining work.

## Quick start (CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

kaggle datasets download -d feyzazkefe/trashnet -p data/trash --unzip
python train.py --data trash --root data/trash/dataset-resized \
                --steps 900 --batch 64 --n 2500 --out runs/trash.pt
python compare.py --data trash --root data/trash/dataset-resized --ckpt runs/trash.pt

python export_c.py   runs/trash.pt --root data/trash/dataset-resized
python export_web.py runs/trash.pt --root data/trash/dataset-resized
make test            # the generated C must match PyTorch (0.9986 cosine, same class)
```

`--data synth` runs without any download; `--data cifar` and `--data stl` are also wired.

## Web demo

`docs/` is two static pages (GitHub Pages serves this folder), no build step and no framework:

- [`docs/index.html`](docs/index.html) — the image side: one photograph, the nearest caption
  prototypes, and a frame switcher that includes frames the model gets wrong.
- [`docs/chat.html`](docs/chat.html) — the text side: type a sentence, get the frames nearest to it.

```bash
python -m http.server 8000     # then open http://localhost:8000/docs/
```

`docs/engine.js` dequantises the same int8 weights and runs the same convolutions in JavaScript:
~30 ms per frame, against 471 ms on the board. `docs/mockups/` holds the six design directions that
were compared before the final pair and is not part of the product.

## Flash it

```bash
cd esp32
idf.py set-target esp32s3          # menuconfig: PSRAM octal, 80 MHz
idf.py -p /dev/tty.usbmodem* flash monitor
```

```json
{"class":"plastic","score":0.412,"ms":471}
```

Pin map in [`firmware/app_main.c`](firmware/app_main.c) is the ESP32-S3-CAM / Freenove layout.

## Layout

```
data_trash.py    TrashNet loader        data_stl.py    STL-10 loader
data_real.py     CIFAR-10 loader        synth.py       procedural world, no download
model.py         image CNN, text encoder, invariance + SIGReg
train.py         two encoders separately, Procrustes fit, CCA, zero-shot
compare.py       the three arms above, one table
export_c.py      int8 quantisation and every generated C file
export_web.py    the same weights for the browser
firmware/        prh_infer.c (runtime), app_main.c (camera + serial), generated weights
esp32/           ESP-IDF project wrapper
host/            test_infer.c — proves the C matches PyTorch
docs/            static demo pages + engine.js (served by GitHub Pages)
docs/            design.md, hardware.md, use-case.md
kaggle/          dataset + kernel packaging for a GPU run
```

## Honest limits

- **48 % is not a product.** Six classes, one dataset, one seed, 43 k parameters. It is a
  demonstration that the mechanism fits in 59 KiB, not that it is accurate.
- **The vocabulary is the boundary.** A class the text encoder cannot place will not be recognised,
  whatever the vision side does.
- **The image side uses multi-crop views**, the paper's image-JEPA arm, not two photographs of one
  object. With one photo per object that is the only option; it is a weaker positive pair.
- **One chosen dataset, not a benchmark.** TrashNet is easy and its backgrounds are clean; a camera
  in a kitchen will find harder frames.

## Licence

MIT for the code. Datasets and pretrained models keep their own terms.

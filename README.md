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
| plain supervised, same CNN end to end | **58.1 %** | all | 3.6 min |
| the paper's method, classes as sentences | **46.9 %** | **0** | 4.0 min |
| the paper's method + 20 labelled photos per class | **49.2 %** | 120 | 0.1 min |
| frozen encoder + linear classifier (probe) | **55.7 %** | all | 10 s |

Chance is 16.7 %. Same encoder (42,960 parameters), same 900 optimisation steps, disjoint train and
test photographs (1,895 / 632). The last row is the diagnostic: it freezes the encoder and trains only
a linear head, so it measures how much class information the encoder actually holds.

If your classes are fixed and your labels exist, use supervised learning. This repository is for the
case where they are not, and for the question of whether the theory survives at 43 k parameters.

## Is the model too small, or is the method unsuitable?

Both, and the probe separates them. A linear classifier on the **frozen** encoder gets 55.7 %, while
the same encoder end-to-end supervised gets 58.1 %: the representation is already close to what a
trained classifier can extract, so the 43 k encoder — not the self-supervised objective — is what
caps the accuracy near 58 %.

The paper's route then gives up another 9 points (55.7 → 46.9 %), because a sentence prototype is a
cruder readout than a trained head. That is the method's cost at this scale, and it does not
disappear with more parameters.

Scaling the encoder is therefore only half an answer: 1,895 training photographs cannot feed
millions of parameters. The honest next step is both — a wider CNN *and* a larger corpus
(STL-10's 5,000, or COCO), which is exactly what `kaggle/` is packaged for:

```bash
./kaggle/pack.sh --push     # PRH_CH=3,48,96,128,192 → ~900 k parameters, GPU
```

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

python export_c.py runs/trash.pt --root data/trash/dataset-resized
make test            # the generated C must match PyTorch (0.9986 cosine, same class)
```

`--data synth` runs without any download; `--data cifar` and `--data stl` are also wired.

## Flash it

```bash
cd esp32
idf.py set-target esp32s3          # menuconfig: PSRAM octal, 80 MHz
idf.py -p /dev/tty.usbmodem* flash monitor
```

The board raises its own Wi-Fi network — no credentials to configure:

```
join  prh-tiny / prhtiny123
open  http://192.168.4.1/
```

The page shows the frame the camera just captured, side by side with the class the model assigned
to it, so you can see the input and the answer together. The serial port prints the same thing:

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
firmware/        prh_infer.c (runtime), app_main.c (camera + serial), generated weights
esp32/           ESP-IDF project wrapper
host/            test_infer.c — proves the C matches PyTorch

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

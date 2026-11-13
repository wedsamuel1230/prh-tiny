# The use case: classes you can add by writing a sentence

## The usual ESP32-CAM classifier

Collect photos of every class you care about, label them, train a CNN, flash it. The class list is
baked into the output layer. Adding "blue triangle" means collecting more photos, retraining,
re-flashing — and if you also want "purple triangle", you do it again.

## What changes once two modalities are aligned

The image encoder learns a *world* representation from unlabelled scenes. The text encoder learns
the same representation from captions. One rotation joins them. After that, a class is **a sentence**:

```
"a small red circle"  ──► text encoder ──► 256-d prototype
```

and classification is the cosine between the image's rotated embedding and the prototypes. No
gradient step, no images, no labels.

Adding a class costs one string. Deleting one costs a `rm`. The model on the flash never changes.

## What that is worth on a microcontroller

| | supervised CNN | this |
|---|---|---|
| data per new class | 50–500 labelled images | one sentence |
| retraining to add a class | yes, then re-flash | no |
| what ships | weights + label list | weights + 32 kB rotation + prototypes |
| failure mode | confidently wrong on unseen classes | weak alignment shows up as a low score |

The last row matters: `prh_classify()` returns the cosine, so a scene that belongs to none of the
classes gives a low score instead of a confident guess. On a sensor that runs unattended for weeks,
that is the property you actually want.

## The demo built here

24 classes: four shapes × six colours, named in English with synonyms ("crimson disk", "gold
wedge"). The camera sees a printed card; the serial port prints

```json
{"class":"red circle","score":0.71,"ms":14}
```

Nothing about the card was labelled. The image encoder trained on renders, the text encoder on
sentences, the rotation on 512 pairs, and the 24 prototypes on sentences alone.

## Where the idea stops

- **The vocabulary is the boundary.** A class the text encoder cannot place relative to the others
  will not be recognised, however good the image encoder is.
- **Alignment quality is the ceiling.** If the image encoder latches onto brightness and the
  captions never mention brightness, the two spaces disagree and every class suffers.
- **Fine-grained classes need a bigger encoder.** 47 kB will separate circles from crosses, not one
  bird species from another. The repo's export path does not change when you scale up; the flash
  budget does.

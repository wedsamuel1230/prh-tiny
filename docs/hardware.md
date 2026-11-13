# Hardware and budget

## Board

ESP32-S3-**N16R8**: 16 MB flash, 8 MB octal PSRAM, 512 KB SRAM, dual-core Xtensa LX7 at 240 MHz,
single-precision FPU, int8/int16 vector instructions, no NPU. Any S3 devkit with an OV2640 works;
the pin map in `firmware/app_main.c` is the ESP32-S3-CAM / Freenove one — change it for yours.

## What actually goes on the flash

| item | size (int8) | note |
|---|---|---|
| image encoder, 43k parameters | 46 KiB | three 3×3 stride-2 convs + linear, NHWC |
| quantisation constants | 1 KiB | per-channel gain + bias |
| rotation R (64→256) | 16 KiB | the paper's only deployable artefact |
| class prototypes (24 × 256) | 6 KiB | int8 |
| **total model data** | **~64 KiB** | 0.5 % of the 13 MiB usable flash |

The second encoder (text) never ships: the prototypes already live in its space.

## What the board actually reports

```
I (785) camera: Detected OV2640 camera
I (885) cam_hal: cam config ok
I (945) prh: camera up; output width 64, 108 KiB of int8 activation buffers
{"class":"purple circle","score":0.568,"ms":471}
```

## RAM

| buffer | size |
|---|---|
| two int8 ping-pong activations | 2 × 36 KiB = 72 KiB |
| RGB888 frame 96×96 | 27 KiB (PSRAM) |
| camera double buffer (RGB565 96×96) | 2 × 18 KiB (PSRAM) |
| prototypes / rotation scratch | 4 KiB |

All of it fits in the 512 KiB SRAM; PSRAM is only used for the camera frames.

## Compute

5.6 M multiply-accumulates per frame (three convs + linear). **Measured on the board: 471 ms**,
i.e. ~12 MMAC/s. That is what a naive scalar C convolution gets; esp-dl's int8 SIMD kernels reach
≈ 0.77 GMAC/s on the same chip (MobileNetV2 int8 224×224 at 389.7 ms for 300 M MAC), which would
put this model near **10 ms**. The gap is the kernel, not the model: reordering the weight layout
to `[ky][kx][oc][ic]` and using the LX7 SIMD instructions is the next optimisation, and it is worth
doing only once the camera path is understood — capture plus RGB565 conversion already costs more
than the network does.

## Power

At 240 MHz the S3 draws roughly 100 mW with the camera idle. One inference every second is a
negligible duty cycle for a battery build; the camera dominates the energy budget, not the model.

#pragma once

#include <stdint.h>

/* rgb: PRH_IMG * PRH_IMG * 3 bytes, RGB888, no stride padding.
 * emb: PRH_D floats (the image modality's embedding). */
void prh_image_embed(const uint8_t *rgb, float *emb);

/* emb -> the text modality's space (PRH_DB floats). */
void prh_align_to_text(const float *x, float *y);

/* Nearest prototype by cosine. Returns the class index; name/score may be NULL. */
int prh_classify(const float *emb, const char **name, float *score);

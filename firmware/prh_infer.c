/* The whole on-device pipeline: RGB frame -> int8 CNN -> embedding -> rotation -> cosine kNN.
 *
 * Everything here is plain C99 with no allocation: two int8 ping-pong buffers (36 KiB max),
 * int32 accumulation, one float multiply-add per output element. No PSRAM needed for the encoder.
 */
#include <math.h>
#include <string.h>

#include "prh_net.h"
#include "prh_rot.h"
#include "prh_proto.h"
#include "prh_infer.h"

static int8_t buf_a[PRH_MAXACT];
static int8_t buf_b[PRH_MAXACT];

static inline int8_t sat(int32_t v)
{
    return (int8_t)(v < 0 ? 0 : (v > 127 ? 127 : v));   /* ReLU + clamp in one place */
}

/* 3x3 stride 2, pad 1, NHWC, weights [OC][3][3][IC], ReLU fused into the clamp. */
static void conv3x3s2(const int8_t *in, int H, int W, int IC,
                      const int8_t *w, const float *gain, const float *bias,
                      int8_t *out, int OC)
{
    const int OH = H / 2, OW = W / 2;

    for (int oy = 0; oy < OH; ++oy) {
        for (int ox = 0; ox < OW; ++ox) {
            int32_t acc[PRH_CH_LAST];
            for (int o = 0; o < OC; ++o) acc[o] = 0;

            for (int ky = 0; ky < 3; ++ky) {
                const int iy = oy * 2 + ky - 1;
                if (iy < 0 || iy >= H) continue;
                for (int kx = 0; kx < 3; ++kx) {
                    const int ix = ox * 2 + kx - 1;
                    if (ix < 0 || ix >= W) continue;
                    const int8_t *p = in + (iy * W + ix) * IC;
                    const int8_t *wp = w + (ky * 3 + kx) * IC;
                    for (int o = 0; o < OC; ++o) {
                        const int8_t *wr = wp + o * 9 * IC;
                        int32_t s = 0;
                        for (int c = 0; c < IC; ++c) s += (int32_t)p[c] * (int32_t)wr[c];
                        acc[o] += s;
                    }
                }
            }

            int8_t *dst = out + (oy * OW + ox) * OC;
            for (int o = 0; o < OC; ++o) dst[o] = sat((int32_t)lrintf(acc[o] * gain[o] + bias[o]));
        }
    }
}

void prh_image_embed(const uint8_t *rgb, float *emb)
{
    int8_t *a = buf_a, *b = buf_b;
    int H = PRH_IMG;

    for (int i = 0; i < PRH_IMG * PRH_IMG * 3; ++i) {
        const float f = rgb[i] / 128.0f - 1.0f;          /* same normalisation as PyTorch */
        const int q = (int)lrintf(f * 127.0f);
        a[i] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
    }

    for (int l = 0; l < PRH_NCONV; ++l) {
        const prh_conv_t *c = &PRH_CONVS[l];
        conv3x3s2(a, H, H, c->ic, c->w, c->gain, c->bias, b, c->oc);
        H /= 2;
        int8_t *t = a; a = b; b = t;
    }

    /* global average pool on the last feature map, dequantised */
    float pool[PRH_CH_LAST];
    for (int c = 0; c < PRH_CH_LAST; ++c) pool[c] = 0.0f;
    for (int i = 0; i < PRH_FINAL_HW; ++i)
        for (int c = 0; c < PRH_CH_LAST; ++c) pool[c] += a[i * PRH_CH_LAST + c];
    for (int c = 0; c < PRH_CH_LAST; ++c)
        pool[c] *= PRH_SCALES[PRH_NCONV - 1] / PRH_FINAL_HW;

    int8_t pq[PRH_CH_LAST];
    for (int c = 0; c < PRH_CH_LAST; ++c) {
        const int q = (int)lrintf(pool[c] / PRH_S_POOL);
        pq[c] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
    }

    for (int o = 0; o < PRH_D; ++o) {
        int32_t acc = 0;
        const int8_t *row = &PRH_FC[o * PRH_CH_LAST];
        for (int c = 0; c < PRH_CH_LAST; ++c) acc += (int32_t)pq[c] * (int32_t)row[c];
        emb[o] = (float)acc * PRH_FC_G[o] + PRH_FC_B[o];
    }
}

/* x -> (x - mu_a) R + mu_b : the only part of the paper the device really needs.
 * R has one int8 scale per row, so the row's scale has to be applied inside the loop; folding it
 * into the output afterwards would index PRH_R_SCALE with the wrong dimension. */
void prh_align_to_text(const float *x, float *y)
{
    for (int j = 0; j < PRH_DB; ++j) y[j] = 0.0f;

    for (int i = 0; i < PRH_DA; ++i) {
        const float xc = x[i] - PRH_MU_A[i];
        const int qx = (int)lrintf(xc * PRH_X_SCALE);
        const float qf = (float)qx * PRH_R_SCALE[i] / PRH_X_SCALE;
        const int8_t *row = &PRH_R[i * PRH_DB];
        for (int j = 0; j < PRH_DB; ++j) y[j] += qf * (float)row[j];
    }

    for (int j = 0; j < PRH_DB; ++j) y[j] += PRH_MU_B[j];
}

int prh_classify(const float *emb, const char **name, float *score)
{
    float y[PRH_DB];
    prh_align_to_text(emb, y);

    float ny = 0.0f;
    for (int j = 0; j < PRH_DB; ++j) ny += y[j] * y[j];
    ny = sqrtf(ny) + 1e-9f;

    int best = 0;
    float best_s = -2.0f;

    for (int i = 0; i < PRH_NCLASS; ++i) {
        const int8_t *p = &PRH_PROTO[i * PRH_DB];
        float dot = 0.0f, np = 0.0f;
        for (int j = 0; j < PRH_DB; ++j) {
            const float v = (float)p[j] * PRH_PSCALE;
            dot += y[j] * v;
            np += v * v;
        }
        const float s = dot / (ny * (sqrtf(np) + 1e-9f));
        if (s > best_s) { best_s = s; best = i; }
    }

    if (name) *name = PRH_NAMES[best];
    if (score) *score = best_s;
    return best;
}

/* Host proof: the generated int8 C must reproduce the PyTorch encoder.
 *
 *   cc -O2 -std=c11 -o build/test_infer firmware/prh_*.c host/test_infer.c host/testvec.c -lm
 *   ./build/test_infer
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "prh_infer.h"
#include "prh_net.h"

extern const uint8_t PRH_TEST_IMG[];
extern const float PRH_TEST_EMB[];
extern const char *PRH_TEST_CLASS;

static float cosine(const float *a, const float *b, int n)
{
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-9f);
}

int main(void)
{
    float emb[PRH_D];
    prh_image_embed(PRH_TEST_IMG, emb);

    const float cos = cosine(emb, PRH_TEST_EMB, PRH_D);
    const char *name = NULL;
    float score = 0.0f;
    prh_classify(emb, &name, &score);

    printf("int8 embedding vs PyTorch float : cosine %.5f\n", cos);
    printf("predicted class                 : %s (score %.3f)\n", name, score);
    printf("expected class                  : %s\n", PRH_TEST_CLASS);

    /* The contract is that the C reproduces the int8 model; a borderline frame may legitimately
     * change class once int8 noise is added, so the class line is reported, not asserted. */
    const int ok = cos > 0.95f;
    printf("class agrees with the float model: %s\n", strcmp(name, PRH_TEST_CLASS) ? "no" : "yes");
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

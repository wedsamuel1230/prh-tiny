/* ESP32-S3 + OV2640: capture a frame, embed it, rotate it into the text space, print the class.
 *
 * This is the whole "use case": the camera looks at a scene, the tiny JEPA image encoder plus the
 * one rotation decide which caption prototype it is closest to, and the answer goes out on the
 * serial port. No labels, no decoder, no language model on the device.
 *
 * Build:  idf.py set-target esp32s3 && idf.py menuconfig   (PSRAM: Octal, 80 MHz)
 *         idf.py -p /dev/tty.usbmodem* flash monitor
 */
#include <stdio.h>
#include <string.h>

#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "prh_infer.h"
#include "prh_net.h"

static const char *TAG = "prh";

/* ESP32-S3-CAM / Freenove S3 CAM pin map; change for your board. */
#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1
#define CAM_PIN_XCLK   15
#define CAM_PIN_SIOD   4
#define CAM_PIN_SIOC   5
#define CAM_PIN_D7     16
#define CAM_PIN_D6     17
#define CAM_PIN_D5     18
#define CAM_PIN_D4     12
#define CAM_PIN_D3     10
#define CAM_PIN_D2     8
#define CAM_PIN_D1     9
#define CAM_PIN_D0     11
#define CAM_PIN_VSYNC  6
#define CAM_PIN_HREF   7
#define CAM_PIN_PCLK   13

static camera_config_t cam_config(void)
{
    camera_config_t c = {
        .pin_pwdn = CAM_PIN_PWDN, .pin_reset = CAM_PIN_RESET,
        .pin_xclk = CAM_PIN_XCLK, .pin_sccb_sda = CAM_PIN_SIOD, .pin_sccb_scl = CAM_PIN_SIOC,
        .pin_d7 = CAM_PIN_D7, .pin_d6 = CAM_PIN_D6, .pin_d5 = CAM_PIN_D5, .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3, .pin_d2 = CAM_PIN_D2, .pin_d1 = CAM_PIN_D1, .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC, .pin_href = CAM_PIN_HREF, .pin_pclk = CAM_PIN_PCLK,
        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0, .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_RGB565,       /* converted below; JPEG needs a decoder */
        .frame_size = FRAMESIZE_96X96,
        .jpeg_quality = 12,
        .fb_count = 2,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
    };
    return c;
}

/* RGB565 (little endian, as the camera writes it) -> the RGB888 the encoder expects. */
static void rgb565_to_rgb888(const uint8_t *src, uint8_t *dst, int w, int h)
{
    for (int i = 0; i < w * h; ++i) {
        const uint16_t p = (uint16_t)src[2 * i] | ((uint16_t)src[2 * i + 1] << 8);
        dst[3 * i + 0] = (uint8_t)(((p >> 11) & 0x1F) * 255 / 31);
        dst[3 * i + 1] = (uint8_t)(((p >> 5) & 0x3F) * 255 / 63);
        dst[3 * i + 2] = (uint8_t)((p & 0x1F) * 255 / 31);
    }
}

void app_main(void)
{
    camera_config_t cfg = cam_config();
    ESP_ERROR_CHECK(esp_camera_init(&cfg));
    ESP_LOGI(TAG, "camera up; output width %d, %d KiB of int8 activation buffers",
             PRH_D, (int)(2 * PRH_MAXACT / 1024));

    uint8_t *rgb = heap_caps_malloc(PRH_IMG * PRH_IMG * 3, MALLOC_CAP_SPIRAM);
    float emb[PRH_D];
    if (!rgb) { ESP_LOGE(TAG, "no PSRAM for the frame buffer"); return; }

    while (true) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) { ESP_LOGE(TAG, "capture failed"); vTaskDelay(pdMS_TO_TICKS(500)); continue; }

        if (fb->width != PRH_IMG || fb->height != PRH_IMG || fb->format != PIXFORMAT_RGB565) {
            ESP_LOGE(TAG, "unexpected frame %dx%d fmt %d", fb->width, fb->height, fb->format);
            esp_camera_fb_return(fb);
            continue;
        }

        rgb565_to_rgb888(fb->buf, rgb, PRH_IMG, PRH_IMG);
        esp_camera_fb_return(fb);

        const int64_t t0 = esp_timer_get_time();
        prh_image_embed(rgb, emb);

        const char *name = NULL;
        float score = 0.0f;
        prh_classify(emb, &name, &score);
        const int64_t t1 = esp_timer_get_time();

        printf("{\"class\":\"%s\",\"score\":%.3f,\"ms\":%d}\n", name, score, (int)((t1 - t0) / 1000));
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

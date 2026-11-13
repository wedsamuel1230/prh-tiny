/* ESP32-S3 + OV2640: capture, embed, rotate, classify — and let you watch it happen.
 *
 * The board raises its own Wi-Fi network, so nothing needs configuring:
 *
 *     connect to  prh-tiny / prhtiny123
 *     open        http://192.168.4.1/
 *
 * The page shows the exact frame the model saw and what it called it. The serial port prints the
 * same thing as JSON, one line per frame.
 */
#include <stdio.h>
#include <string.h>

#include "esp_camera.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "prh_infer.h"
#include "prh_net.h"

static const char *TAG = "prh";

#define AP_SSID "prh-tiny"
#define AP_PASS "prhtiny123"
#define PREVIEW_W 240
#define PREVIEW_H 240

/* ESP32-S3-CAM / Freenove pin map; change for your board. */
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

/* latest frame, RGB565 little endian, read by the HTTP handlers.
 * 115 KiB: it lives in PSRAM, or the Wi-Fi stack and this buffer do not both fit in DRAM. */
static uint16_t *s_frame;
static int s_w = PREVIEW_W, s_h = PREVIEW_H;
static volatile uint32_t s_frame_seq;
static char s_class[32] = "…";
static float s_score;
static int s_ms;

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
        .pixel_format = PIXFORMAT_RGB565,
        .frame_size = FRAMESIZE_240X240,
        .fb_count = 2,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
    };
    return c;
}

/* RGB565 -> the RGB888 the encoder expects, nearest-neighbour to PRH_IMG. */
static void to_rgb96(const uint8_t *src, int w, int h, uint8_t *dst)
{
    for (int y = 0; y < PRH_IMG; ++y) {
        const int sy = y * h / PRH_IMG;
        for (int x = 0; x < PRH_IMG; ++x) {
            const int sx = x * w / PRH_IMG;
            const uint16_t p = (uint16_t)src[2 * (sy * w + sx)] |
                               ((uint16_t)src[2 * (sy * w + sx) + 1] << 8);
            uint8_t *o = dst + 3 * (y * PRH_IMG + x);
            o[0] = (uint8_t)(((p >> 11) & 0x1F) * 255 / 31);
            o[1] = (uint8_t)(((p >> 5) & 0x3F) * 255 / 63);
            o[2] = (uint8_t)((p & 0x1F) * 255 / 31);
        }
    }
}

static void capture_task(void *arg)
{
    const size_t rgb_len = PRH_IMG * PRH_IMG * 3;
    uint8_t *rgb = heap_caps_malloc(rgb_len, MALLOC_CAP_SPIRAM);
    float emb[PRH_D];
    if (!rgb) {
        ESP_LOGE(TAG, "no PSRAM for the RGB buffer");
        vTaskDelete(NULL);
    }

    while (true) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) { vTaskDelay(pdMS_TO_TICKS(200)); continue; }

        s_w = fb->width;
        s_h = fb->height;
        const size_t bytes = (size_t)fb->width * fb->height * 2;
        memcpy(s_frame, fb->buf, bytes < PREVIEW_W * PREVIEW_H * 2 ? bytes : PREVIEW_W * PREVIEW_H * 2);
        to_rgb96(fb->buf, fb->width, fb->height, rgb);
        esp_camera_fb_return(fb);
        s_frame_seq++;

        const int64_t t0 = esp_timer_get_time();
        prh_image_embed(rgb, emb);
        const char *name = NULL;
        float score = 0.0f;
        prh_classify(emb, &name, &score);
        const int64_t t1 = esp_timer_get_time();

        strncpy(s_class, name ? name : "?", sizeof(s_class) - 1);
        s_score = score;
        s_ms = (int)((t1 - t0) / 1000);

        printf("{\"class\":\"%s\",\"score\":%.3f,\"ms\":%d}\n", s_class, s_score, s_ms);
        fflush(stdout);
        vTaskDelay(pdMS_TO_TICKS(120));       /* ~7 fps of inference, page updates slower */
    }
}

static const char PAGE[] =
"<!DOCTYPE html><html><head><meta charset=utf-8>"
"<meta name=viewport content='width=device-width,initial-scale=1'>"
"<title>prh-tiny</title><style>"
"body{margin:0;background:#111;color:#eee;font:14px ui-monospace,monospace;text-align:center}"
"img{width:min(92vw,440px);image-rendering:pixelated;margin-top:18px;border-radius:4px}"
"h1{font:16px ui-sans-serif,system-ui;font-weight:500;margin:18px 0 4px}"
".s{color:#8ec; font-size:26px;margin:10px 0 2px}"
".m{color:#888;font-size:12px}"
"</style></head><body>"
"<h1>prh-tiny · what the camera sees</h1>"
"<img id=f alt='camera frame'>"
"<div class=s id=c>…</div><div class=m id=v></div>"
"<script>"
"const f=document.getElementById('f');"
"f.src='/frame.jpg';setInterval(()=>{f.src='/frame.jpg?t='+Date.now()},1200);"
"async function tick(){try{const r=await(await fetch('/api')).json();"
"document.getElementById('c').textContent=r.class;"
"document.getElementById('v').textContent=r.score.toFixed(3)+' · '+r.ms+' ms · '+r.frame+' frames';"
"}catch(e){}setTimeout(tick,1000)}tick();"
"</script></body></html>";

static esp_err_t h_root(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, PAGE, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t h_api(httpd_req_t *req)
{
    char buf[128];
    int n = snprintf(buf, sizeof(buf), "{\"class\":\"%s\",\"score\":%.3f,\"ms\":%d,\"frame\":%u}",
                     s_class, s_score, s_ms, (unsigned)s_frame_seq);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, buf, n);
}

/* 240x240 RGB565 as a BMP, so any browser can draw it without a decoder. */
static esp_err_t h_frame(httpd_req_t *req)
{
    const int w = s_w, h = s_h;
    const int row = w * 3;
    const int pad = (4 - (row % 4)) % 4;
    const int data = (row + pad) * h;
    const int size = 54 + data;

    uint8_t hdr[54] = {0};
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &size, 4);
    const int off = 54; memcpy(hdr + 10, &off, 4);
    const int dib = 40; memcpy(hdr + 14, &dib, 4);
    memcpy(hdr + 18, &w, 4); memcpy(hdr + 22, &h, 4);
    hdr[26] = 1; hdr[28] = 24;

    httpd_resp_set_type(req, "image/bmp");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    esp_err_t err = httpd_resp_send_chunk(req, (const char *)hdr, 54);
    if (err != ESP_OK) return err;

    static uint8_t line[PREVIEW_W * 3 + 4];
    for (int y = h - 1; y >= 0; --y) {                 /* BMP rows are bottom-up */
        for (int x = 0; x < w; ++x) {
            const uint16_t p = s_frame[y * w + x];
            line[3 * x + 0] = (uint8_t)((p & 0x1F) * 255 / 31);
            line[3 * x + 1] = (uint8_t)(((p >> 5) & 0x3F) * 255 / 63);
            line[3 * x + 2] = (uint8_t)(((p >> 11) & 0x1F) * 255 / 31);
        }
        for (int i = 0; i < pad; ++i) line[row + i] = 0;
        err = httpd_resp_send_chunk(req, (const char *)line, row + pad);
        if (err != ESP_OK) return err;
    }
    return httpd_resp_send_chunk(req, NULL, 0);
}

static void start_http(void)
{
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.stack_size = 6144;
    cfg.lru_purge_enable = true;
    cfg.max_open_sockets = 3;      /* CONFIG_LWIP_MAX_SOCKETS must exceed this */
    httpd_handle_t srv = NULL;
    esp_err_t err = httpd_start(&srv, &cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "httpd did not start: %s (0x%x)", esp_err_to_name(err), err);
        return;
    }
    httpd_uri_t root = {.uri = "/", .method = HTTP_GET, .handler = h_root};
    httpd_uri_t api = {.uri = "/api", .method = HTTP_GET, .handler = h_api};
    httpd_uri_t img = {.uri = "/frame.jpg", .method = HTTP_GET, .handler = h_frame};
    httpd_register_uri_handler(srv, &root);
    httpd_register_uri_handler(srv, &api);
    httpd_register_uri_handler(srv, &img);
}

static void start_ap(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t ic = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&ic));

    wifi_config_t wc = {0};
    strncpy((char *)wc.ap.ssid, AP_SSID, sizeof(wc.ap.ssid));
    wc.ap.ssid_len = strlen(AP_SSID);
    strncpy((char *)wc.ap.password, AP_PASS, sizeof(wc.ap.password));
    wc.ap.channel = 1;
    wc.ap.max_connection = 4;
    wc.ap.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "join Wi-Fi '%s' (password '%s') then open http://192.168.4.1/", AP_SSID, AP_PASS);
}

void app_main(void)
{
    s_frame = heap_caps_malloc(PREVIEW_W * PREVIEW_H * 2, MALLOC_CAP_SPIRAM);
    if (!s_frame) {
        ESP_LOGE(TAG, "no PSRAM for the preview buffer");
        return;
    }

    camera_config_t cfg = cam_config();
    ESP_ERROR_CHECK(esp_camera_init(&cfg));
    ESP_LOGI(TAG, "camera up: %dx%d, %d KiB of int8 activation buffers",
             PREVIEW_W, PREVIEW_H, (int)(2 * PRH_MAXACT / 1024));

    start_ap();
    start_http();

    xTaskCreatePinnedToCore(capture_task, "capture", 8192, NULL, 5, NULL, 0);
}

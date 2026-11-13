idf_component_register(
    SRCS "app_main.c" "prh_infer.c" "prh_net.c" "prh_rot.c" "prh_proto.c"
    INCLUDE_DIRS "."
    REQUIRES esp32-camera esp_timer heap
)

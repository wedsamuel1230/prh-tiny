CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -ffp-contract=off
INC     := -Ifirmware -Ihost

SRC := firmware/prh_infer.c firmware/prh_net.c firmware/prh_rot.c firmware/prh_proto.c
HOST := host/test_infer.c host/testvec.c

.PHONY: all train export test clean

all: test

train:
	python train.py --views good --steps 1500 --batch 64 --n 2500

export:
	python export_c.py runs/tiny.pt

test: build/test_infer
	./build/test_infer

build/test_infer: $(SRC) $(HOST)
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) $(HOST) -lm

clean:
	rm -rf build

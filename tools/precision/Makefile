# C-side Qwen3 logits dumper for tools/precision/compare-qwen3.py.
# Building this also refreshes the root soft engine library.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic
override CFLAGS += -std=c11
LDLIBS := -lm
ENGINE_LIB := $(ROOT)/build/soft/libbnn-engine.a

.PHONY: all clean

all: dump-logits

dump-logits: dump-logits.c
	$(MAKE) -C $(ROOT) ACCE=soft
	$(CC) $(CFLAGS) -I$(ROOT)/bnn-sw-v2 -I$(ROOT)/bnn-accelerator $< $(ENGINE_LIB) $(LDLIBS) -o $@

clean:
	$(RM) dump-logits

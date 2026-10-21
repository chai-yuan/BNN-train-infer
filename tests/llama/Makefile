# Llama-2 end-to-end regression: export -> inference -> reference check.
#
#   make test
#   make ACCE=verilator test
#
# The exporter/test may need a Python environment with its dependencies.
# Override with PYTHON=/absolute/path/to/.venv/bin/python3 if needed.

TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(TEST_DIR)/../..)

ACCE ?= soft
PYTHON ?= python3

ifeq ($(filter $(ACCE),soft verilator),)
$(error Unsupported ACCE '$(ACCE)'; use soft or verilator)
endif

.DEFAULT_GOAL := test

.PHONY: build test

build:
	$(MAKE) -C $(ROOT) ACCE=$(ACCE)

test: build
	$(PYTHON) $(TEST_DIR)/test_llama.py

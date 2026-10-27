# Standalone C11 soft accelerator API v2 regressions; no root build or RTL required.
#   make test
#   make test CC='ccache gcc' CFLAGS='-O1 -g -fsanitize=address,undefined'

TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

PYTHON ?= python3
CC ?= cc
CFLAGS ?=

.DEFAULT_GOAL := test

.PHONY: test

test:
	$(PYTHON) "$(TEST_DIR)/test_accelerator.py" --cc "$(CC)" --cflags="$(CFLAGS)"

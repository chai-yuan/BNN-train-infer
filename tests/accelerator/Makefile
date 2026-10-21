# Accelerator API and soft/RTL differential regressions.
# The differential test always needs the ACCE=verilator archive; the soft
# runner is compiled directly from bnn-accelerator/soft-bnn-accelerator.c.
#
#   make test
#   make waves        # keep VCDs under build/waves/accelerator
#   make waves OUTPUT_DIR=build/my-waves

TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(TEST_DIR)/../..)

PYTHON ?= python3
CC ?= cc
CXX ?= c++
VERILATOR ?= verilator

LIBRARY := $(ROOT)/build/verilator/libbnn-engine.a
OUTPUT_DIR ?= $(ROOT)/build/waves/accelerator
TEST_ARGS := --library "$(LIBRARY)" --cc "$(CC)" --cxx "$(CXX)" --verilator "$(VERILATOR)"

ifeq ($(filter /%,$(OUTPUT_DIR)),)
OUTPUT_DIR := $(ROOT)/$(OUTPUT_DIR)
endif

.DEFAULT_GOAL := test

.PHONY: build test waves

build:
	$(MAKE) -C $(ROOT) ACCE=verilator

test: build
	$(PYTHON) $(TEST_DIR)/test_accelerator.py $(TEST_ARGS)

waves: build
	mkdir -p "$(OUTPUT_DIR)"
	$(PYTHON) $(TEST_DIR)/test_accelerator.py $(TEST_ARGS) --trace-dir "$(OUTPUT_DIR)"

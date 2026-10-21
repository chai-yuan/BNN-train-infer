# Optional VCD waveform regressions through the public BNN API.
#
#   make test          # temporary waveforms, cleaned automatically
#   make waves         # keep waveforms under build/waves
#   make waves OUTPUT_DIR=build/my-waves

TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(TEST_DIR)/../..)

PYTHON ?= python3
CC ?= cc
CXX ?= c++
LIBRARY := $(ROOT)/build/verilator/libbnn-engine.a
OUTPUT_DIR ?= $(ROOT)/build/waves

ifeq ($(filter /%,$(OUTPUT_DIR)),)
OUTPUT_DIR := $(ROOT)/$(OUTPUT_DIR)
endif

.DEFAULT_GOAL := test

.PHONY: build test waves

build:
	$(MAKE) -C $(ROOT) ACCE=verilator

test: build
	$(PYTHON) $(TEST_DIR)/test_verilator_trace.py \
		--library "$(LIBRARY)" --cc "$(CC)" --cxx "$(CXX)"

waves: build
	mkdir -p "$(OUTPUT_DIR)"
	$(PYTHON) $(TEST_DIR)/test_verilator_trace.py \
		--library "$(LIBRARY)" --cc "$(CC)" --cxx "$(CXX)" \
		--output-dir "$(OUTPUT_DIR)"

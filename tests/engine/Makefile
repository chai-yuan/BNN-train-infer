# Engine/CLI regressions: MLP, CNN, GPT, and the public BNNE API.
#
#   make test                         # default ACCE=soft
#   make ACCE=verilator test          # same tests against the RTL backend
#   make api                          # engine API runner only
#   make fixtures FIXTURES=build/fixtures

TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(TEST_DIR)/../..)

ACCE ?= soft
PYTHON ?= python3
CC ?= cc
CXX ?= c++

BINARY := $(ROOT)/bnn-infer
LIBRARY := $(ROOT)/build/$(ACCE)/libbnn-engine.a
FIXTURES ?= $(ROOT)/build/fixtures

ifeq ($(ACCE),soft)
TEST_LINKER := $(CC)
TEST_LDLIBS := -lm
else ifeq ($(ACCE),verilator)
TEST_LINKER := $(CXX)
TEST_LDLIBS := -lm -pthread -latomic
else
$(error Unsupported ACCE '$(ACCE)'; use soft or verilator)
endif

ifeq ($(filter /%,$(FIXTURES)),)
FIXTURES := $(ROOT)/$(FIXTURES)
endif

.DEFAULT_GOAL := test

.PHONY: build test api fixtures

build:
	$(MAKE) -C $(ROOT) ACCE=$(ACCE)

test: build
	$(PYTHON) $(TEST_DIR)/test_mlp.py \
		--binary "$(BINARY)" --library "$(LIBRARY)" \
		--cc "$(CC)" --linker "$(TEST_LINKER)" --ldlibs="$(TEST_LDLIBS)"

api: build
	$(PYTHON) $(TEST_DIR)/test_mlp.py \
		--binary "$(BINARY)" --library "$(LIBRARY)" \
		--cc "$(CC)" --linker "$(TEST_LINKER)" --ldlibs="$(TEST_LDLIBS)" \
		--api-only

fixtures:
	$(PYTHON) $(TEST_DIR)/test_mlp.py --fixtures-only "$(FIXTURES)"

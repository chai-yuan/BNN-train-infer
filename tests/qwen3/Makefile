# Qwen3 / Bonsai Q1_0 differential test.
#
#   make test BONSAI_GGUF=/path/to/Bonsai-1.7B-Q1_0.gguf
#   make generation BONSAI_GGUF=...   # also exercise --ids + KV cache

TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(TEST_DIR)/../..)

PYTHON ?= python3
BONSAI_GGUF ?=
LLAMA_TOKENIZE ?=
BINARY := $(ROOT)/bnn-infer

.PHONY: build test tokenizer generation

build:
	$(MAKE) -C $(ROOT) ACCE=soft

test: build
	$(PYTHON) $(TEST_DIR)/test_qwen3.py --model "$(BONSAI_GGUF)" --binary "$(BINARY)"
	$(PYTHON) $(TEST_DIR)/test_tokenizer.py --model "$(BONSAI_GGUF)" --binary "$(BINARY)" \
		--llama-tokenize "$(LLAMA_TOKENIZE)"

tokenizer: build
	$(PYTHON) $(TEST_DIR)/test_tokenizer.py --model "$(BONSAI_GGUF)" --binary "$(BINARY)" \
		--llama-tokenize "$(LLAMA_TOKENIZE)"

generation: build
	$(PYTHON) $(TEST_DIR)/test_qwen3.py --model "$(BONSAI_GGUF)" --binary "$(BINARY)" --generation

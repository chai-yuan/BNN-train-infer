TOOL_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TOOLS_ROOT := $(abspath $(TOOL_DIR)/..)
PORT ?= 8766

.PHONY: serve test check

serve:
	@printf 'GGUF/LENS: http://127.0.0.1:%s/gguf-lens/\n' "$(PORT)"
	python3 -m http.server "$(PORT)" --bind 127.0.0.1 --directory "$(TOOLS_ROOT)"

test: check
	node --test "$(TOOL_DIR)/test/gguf.test.js"

check:
	node --check "$(TOOL_DIR)/js/gguf.js"
	node --check "$(TOOL_DIR)/js/app.js"
	@echo "JS syntax: OK"

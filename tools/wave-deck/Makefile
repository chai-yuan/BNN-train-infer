TOOL_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PORT ?= 8765
JS := $(wildcard $(TOOL_DIR)/js/*.js)

.PHONY: serve test check

serve:
	@printf 'WAVE/DECK: http://127.0.0.1:%s/\n' "$(PORT)"
	python3 -m http.server "$(PORT)" --bind 127.0.0.1 --directory "$(TOOL_DIR)"

test: check
	node "$(TOOL_DIR)/test/analyze.test.js"

check:
	@for f in $(JS); do node --check "$$f" || exit 1; done
	@echo "JS syntax: OK"

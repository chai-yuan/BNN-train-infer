# Standalone Verilator tests; no dependency on the software/Python build.
#   make -C bnn-hw/bnn_core_test test
#   make -C bnn-hw/bnn_core_test test SEEDS="7 42" RANDOM_CASES=40
#   make -C bnn-hw/bnn_core_test test SEEDS=7 PLUSARGS="+no_throughput"
#   make -C bnn-hw/bnn_core_test lint
# +no_throughput is an explicit diagnostic escape hatch, not the default.
.DEFAULT_GOAL := test

TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(TEST_DIR)/../..)
RTL_DIR := $(ROOT)/bnn-hw/bnn_core
BUILD_DIR := $(ROOT)/build/bnn-core-test
VERILATOR ?= verilator
JOBS ?= 2
SEEDS ?= 1 7 42
RANDOM_CASES ?= 16
MAX_CYCLES ?= 500000
PLUSARGS ?=

# Explicit dependency order. The core has no package dependency.
RTL := $(RTL_DIR)/bnn_ctrl_fsm.sv \
       $(RTL_DIR)/bnn_addertree.sv \
       $(RTL_DIR)/bnn_group_acc.sv \
       $(RTL_DIR)/bnn_out_slice.sv \
       $(RTL_DIR)/bnn_core_top.sv
TB := $(TEST_DIR)/tb_bnn_core.sv
PARAM_TB := $(TEST_DIR)/tb_bnn_core_params.sv
SIM := $(BUILD_DIR)/obj/Vtb_bnn_core
# Warnings remain fatal. These suppressions cover intentional testbench idioms.
VFLAGS := --timing --assert -Wall -Wno-DECLFILENAME -Wno-BLKSEQ \
          -Wno-UNUSEDSIGNAL -Wno-PINCONNECTEMPTY

.PHONY: test test-main test-params lint clean

test: test-main test-params

test-main: $(SIM)
	@set -eu; for seed in $(SEEDS); do \
		printf '\n=== bnn_core seed=%s ===\n' "$$seed"; \
		ulimit -c 0; \
		$(SIM) +seed=$$seed +random_cases=$(RANDOM_CASES) \
			+max_cycles=$(MAX_CYCLES) $(PLUSARGS) \
			> $(BUILD_DIR)/seed-$$seed.log 2>&1 || { \
				cat $(BUILD_DIR)/seed-$$seed.log; exit 1; }; \
		cat $(BUILD_DIR)/seed-$$seed.log; \
	done

$(SIM): $(RTL) $(TB) $(TEST_DIR)/Makefile
	@mkdir -p $(BUILD_DIR)/obj
	$(VERILATOR) --binary $(VFLAGS) --top-module tb_bnn_core \
		--Mdir $(BUILD_DIR)/obj -j $(JOBS) $(RTL) $(TB)

# Build failures are never accepted as evidence of parameter rejection: each
# probe must compile, then terminate nonzero with an RTL assertion diagnostic.
# Widening probes keep otherwise fixed datapath expressions elaboratable.
# Only these invalid configurations suppress WIDTH: their datapath widths are
# deliberately inconsistent, and the runtime parameter guard is the test target.
test-params:
	@mkdir -p $(BUILD_DIR)
	@set -eu; for spec in P_LANES:64 ACT_WIDTH:9 PSUM_WIDTH:17 RES_WIDTH:33; do \
		param=$${spec%:*}; value=$${spec#*:}; \
		out=$(BUILD_DIR)/param-$$param; \
		mkdir -p "$$out"; \
		printf '\n=== unsupported %s=%s must fail at runtime ===\n' "$$param" "$$value"; \
		$(VERILATOR) --binary $(VFLAGS) -Wno-WIDTH --top-module tb_bnn_core_params \
			--Mdir "$$out" -j $(JOBS) -GTEST_$$param=$$value \
			$(RTL) $(PARAM_TB) > "$$out/build.log" 2>&1 || { \
				cat "$$out/build.log"; exit 1; }; \
		ulimit -c 0; \
		if "$$out/Vtb_bnn_core_params" > "$$out/run.log" 2>&1; then \
			cat "$$out/run.log"; \
			printf 'FAIL: RTL accepted unsupported %s=%s\n' "$$param" "$$value"; exit 1; \
		fi; \
		if ! grep -Eq 'Assertion failed.*bnn_(core_top|addertree|group_acc|out_slice)' "$$out/run.log"; then \
			cat "$$out/run.log"; \
			printf 'FAIL: parameter probe exited without an RTL assertion\n'; exit 1; \
		fi; \
		printf 'PASS: %s=%s rejected by RTL\n' "$$param" "$$value"; \
	done

lint:
	$(VERILATOR) --lint-only $(VFLAGS) --top-module tb_bnn_core $(RTL) $(TB)
	$(VERILATOR) --lint-only $(VFLAGS) --top-module tb_bnn_core_params $(RTL) $(PARAM_TB)

clean:
	rm -rf -- $(BUILD_DIR)

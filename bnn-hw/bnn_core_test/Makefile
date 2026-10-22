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
BUILD_DIR ?= $(ROOT)/build/bnn-core-test
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
CONFIG_TB := $(TEST_DIR)/tb_bnn_core_config.sv
SIM := $(BUILD_DIR)/obj/Vtb_bnn_core
CONFIG_SIM := $(BUILD_DIR)/config/Vtb_bnn_core_config
# Warnings remain fatal. These suppressions cover intentional testbench idioms.
VFLAGS := --timing --assert -Wall -Wno-DECLFILENAME -Wno-BLKSEQ \
          -Wno-UNUSEDSIGNAL -Wno-PINCONNECTEMPTY

.PHONY: test test-main test-config test-params lint clean

test: test-main test-config test-params

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

$(CONFIG_SIM): $(RTL) $(CONFIG_TB) $(TEST_DIR)/Makefile
	@mkdir -p $(BUILD_DIR)/config
	$(VERILATOR) --binary $(VFLAGS) --top-module tb_bnn_core_config \
		--Mdir $(BUILD_DIR)/config -j $(JOBS) $(RTL) $(CONFIG_TB)

# All guard scenarios must pass before the accepted-command assertion. Require
# the specific RTL diagnostic as well as the expected nonzero runtime exit.
test-config: $(CONFIG_SIM)
	@set -eu; ulimit -c 0; \
		$(CONFIG_SIM) +probe=0 > $(BUILD_DIR)/config/guards.log 2>&1 || { \
			cat $(BUILD_DIR)/config/guards.log; exit 1; }; \
		cat $(BUILD_DIR)/config/guards.log; \
		for probe in 1 2 3 4 5 6 7 8; do \
			case $$probe in \
				1) diagnostic='cmd_k_blocks must be nonzero' ;; \
				2) diagnostic='cmd_group_div must be nonzero' ;; \
				3) diagnostic='cmd_out_chans must be nonzero' ;; \
				4|5) diagnostic='cmd_k_blocks must be divisible by cmd_group_div' ;; \
				6|7|8) diagnostic='command must contain at most 256 groups' ;; \
			esac; \
			out=$(BUILD_DIR)/config/probe-$$probe.log; \
			if $(CONFIG_SIM) +probe=$$probe > "$$out" 2>&1; then \
				cat "$$out"; printf 'FAIL: illegal command probe %s succeeded\n' "$$probe"; exit 1; \
			fi; \
			if ! grep -Fq '[GUARDS PASS]' "$$out" || \
			   ! grep -Fq "[EXPECT ASSERT] probe=$$probe " "$$out" || \
			   ! grep -Eq 'bnn_ctrl_fsm[.]sv:.*Assertion failed' "$$out" || \
			   ! grep -Fq "$$diagnostic" "$$out" || \
			   grep -Fq '[UNEXPECTED]' "$$out"; then \
				cat "$$out"; printf 'FAIL: probe %s lacked its immediate RTL assertion\n' "$$probe"; exit 1; \
			fi; \
			printf 'PASS: command probe %s rejected at handshake (%s)\n' "$$probe" "$$diagnostic"; \
		done

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
		if ! grep -Eq 'bnn_(core_top|addertree|group_acc|out_slice)[.]sv:.*Assertion failed' "$$out/run.log"; then \
			cat "$$out/run.log"; \
			printf 'FAIL: parameter probe exited without an RTL assertion\n'; exit 1; \
		fi; \
		printf 'PASS: %s=%s rejected by RTL\n' "$$param" "$$value"; \
	done

lint:
	$(VERILATOR) --lint-only $(VFLAGS) --top-module tb_bnn_core $(RTL) $(TB)
	$(VERILATOR) --lint-only $(VFLAGS) --top-module tb_bnn_core_params $(RTL) $(PARAM_TB)
	$(VERILATOR) --lint-only $(VFLAGS) --top-module tb_bnn_core_config $(RTL) $(CONFIG_TB)

clean:
	rm -rf -- $(BUILD_DIR)

VERILATOR ?= verilator
CXX ?= g++
PYTHON ?= python3
JOBS ?= 2
DMA_WIDTH ?= 128
BUILD := build
SIM_DIR := $(abspath .)
FP_PACKAGE := ../bnn_core/bnn_fp_math.sv
CORE_RTL := $(FP_PACKAGE) $(sort $(filter-out $(FP_PACKAGE),$(wildcard ../bnn_core/*.sv)))
HEADERS := $(wildcard include/*.hpp)
VFLAGS := --cc --exe --build --assert --trace -Wall -j $(JOBS)
VFLAGS_COMB := --cc --exe --build --assert -Wall -j $(JOBS)
CXXFLAGS_SIM := -std=c++17 -Wall -Wextra -fno-fast-math -ffp-contract=off -I$(SIM_DIR)/include
DMA_EXE := $(BUILD)/dma-$(DMA_WIDTH)/Vbnn_dma_memory
FP_TEST_EXE := $(BUILD)/fp_math_test/Vfp_math_test
CORE_EXE := $(BUILD)/core/Vbnn_sim_top
CORE_TEST_EXE := $(BUILD)/core-test/Vbnn_sim_top

.PHONY: all test host-test fp-test dma-test negative-test test-widths core core-test core-smoke run lint lint-core clean help FORCE
all: test

$(BUILD):
	mkdir -p $(BUILD)

# Flags are not normal file prerequisites. Keep a content stamp, and discard only
# the affected generated build when flags change so stale objects cannot survive.
FORCE:

$(BUILD)/host_driver_test.config $(BUILD)/dma-$(DMA_WIDTH).config $(BUILD)/fp_math_test.config $(BUILD)/core.config $(BUILD)/core-test.config: FORCE | $(BUILD)
	@printf '%s\n' '$(VERILATOR)' '$(CXX)' '$(VFLAGS)' '$(VFLAGS_COMB)' '$(CXXFLAGS_SIM)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else rm -rf $(basename $@); mv -f $@.tmp $@; fi

$(BUILD)/host_driver_test: host_driver_test.cpp include/host_driver.hpp Makefile $(BUILD)/host_driver_test.config | $(BUILD)
	$(CXX) -std=c++17 -Wall -Wextra -Werror -Iinclude $< -o $@

host-test: $(BUILD)/host_driver_test
	./$(BUILD)/host_driver_test

$(FP_TEST_EXE): $(FP_PACKAGE) fp_math_test.sv fp_math_test.cpp Makefile $(BUILD)/fp_math_test.config | $(BUILD)
	$(VERILATOR) $(VFLAGS_COMB) --top-module fp_math_test --Mdir $(BUILD)/fp_math_test \
		-CFLAGS "$(CXXFLAGS_SIM)" \
		$(FP_PACKAGE) fp_math_test.sv $(SIM_DIR)/fp_math_test.cpp

fp-test: $(FP_TEST_EXE)
	./$(FP_TEST_EXE)

$(DMA_EXE): bnn_dma_memory.sv dma_memory_test.cpp virtual_memory.cpp $(HEADERS) Makefile $(BUILD)/dma-$(DMA_WIDTH).config | $(BUILD)
	$(VERILATOR) $(VFLAGS) --top-module bnn_dma_memory --Mdir $(BUILD)/dma-$(DMA_WIDTH) \
		-GADDR_WIDTH=64 -GDMA_DATA_WIDTH=$(DMA_WIDTH) \
		-CFLAGS "$(CXXFLAGS_SIM) -DTEST_DMA_WIDTH=$(DMA_WIDTH)" \
		bnn_dma_memory.sv $(SIM_DIR)/dma_memory_test.cpp $(SIM_DIR)/virtual_memory.cpp

dma-test: $(DMA_EXE)
	./$(DMA_EXE) --seed 1
	./$(DMA_EXE) --seed 7
	./$(DMA_EXE) --seed 42

negative-test: $(DMA_EXE)
	$(PYTHON) test_protocol_errors.py ./$(DMA_EXE)

test: host-test fp-test dma-test negative-test core-test core-smoke

test-widths:
	$(MAKE) dma-test negative-test DMA_WIDTH=64
	$(MAKE) dma-test negative-test DMA_WIDTH=128
	$(MAKE) dma-test negative-test DMA_WIDTH=256

# Compile the arithmetic package before the modules that import it.
$(CORE_EXE): bnn_sim_top.sv bnn_dma_memory.sv main.cpp virtual_memory.cpp $(HEADERS) $(CORE_RTL) Makefile $(BUILD)/core.config | $(BUILD)
	$(VERILATOR) $(VFLAGS) \
		--top-module bnn_sim_top --Mdir $(BUILD)/core -CFLAGS "$(CXXFLAGS_SIM)" \
		bnn_sim_top.sv bnn_dma_memory.sv $(CORE_RTL) $(SIM_DIR)/main.cpp $(SIM_DIR)/virtual_memory.cpp

core: $(CORE_EXE)

$(CORE_TEST_EXE): bnn_sim_top.sv bnn_dma_memory.sv core_test.cpp virtual_memory.cpp $(HEADERS) $(CORE_RTL) Makefile $(BUILD)/core-test.config | $(BUILD)
	$(VERILATOR) $(VFLAGS) --top-module bnn_sim_top --Mdir $(BUILD)/core-test \
		-CFLAGS "$(CXXFLAGS_SIM)" bnn_sim_top.sv bnn_dma_memory.sv $(CORE_RTL) \
		$(SIM_DIR)/core_test.cpp $(SIM_DIR)/virtual_memory.cpp

core-test: $(CORE_TEST_EXE)
	./$(CORE_TEST_EXE) --seed 1
	./$(CORE_TEST_EXE) --seed 7
	./$(CORE_TEST_EXE) --seed 42

core-smoke: core
	$(PYTHON) test_cli.py ./$(CORE_EXE)

run: core
	./$(CORE_EXE) $(ARGS)

lint:
	$(VERILATOR) --lint-only --assert -Wall --top-module bnn_dma_memory bnn_dma_memory.sv
	$(VERILATOR) --lint-only --assert -Wall --top-module fp_math_test $(FP_PACKAGE) fp_math_test.sv

lint-core:
	$(VERILATOR) --lint-only --assert -Wall --top-module bnn_core_top $(CORE_RTL)
	$(VERILATOR) --lint-only -Wall -DSYNTHESIS --top-module bnn_core_top $(CORE_RTL)
	$(VERILATOR) --lint-only --assert -Wall --top-module bnn_sim_top bnn_sim_top.sv bnn_dma_memory.sv $(CORE_RTL)

clean:
	rm -rf $(BUILD)

help:
	@printf '%s\n' 'make test         Full host/FP/DMA/negative/core/CLI regression suite' \
		'make test-widths  Repeat DMA + protocol negative tests at 64/128/256-bit widths' \
		'make fp-test      Combinational bnn_fp_math arithmetic regression' \
		'make core         Build the actual bnn_core_top CLI simulator' \
		'make core-test    End-to-end instruction/DMA/matrix regression' \
		'make core-smoke   CLI idle, command-file, binary I/O, VCD and error tests' \
		'make run ARGS="--help"  See core simulator CLI' \
		'make lint         Strict DMA-model and fp_math_test lint' \
		'make lint-core    Strict core RTL (sim + SYNTHESIS) and sim-top lint' \
		'make clean        Remove sim/build only'

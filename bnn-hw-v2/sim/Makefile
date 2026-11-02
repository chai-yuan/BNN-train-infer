# Minimal Verilator test flow for the BNN v2 simulation top.
VERILATOR ?= verilator
PYTHON    ?= python3
JOBS      ?= 2
SEED      ?= 1

BUILD     := build
DATA_DIR  := tests/case
TEST_DIR  := $(BUILD)/testdata
SIM       := $(BUILD)/Vbnn_sim_top

FP_PACKAGE := ../bnn_core/bnn_fp_math.sv
CORE_RTL   := $(FP_PACKAGE) $(filter-out $(FP_PACKAGE),$(wildcard ../bnn_core/*.sv))
SIM_RTL    := $(sort $(wildcard sim_rtl/*.sv))

.PHONY: all build test gen run check clean help
.DEFAULT_GOAL := all

all: build
build: $(SIM)

$(SIM): $(SIM_RTL) $(CORE_RTL) simple_test_main.cpp Makefile
	$(VERILATOR) --cc --exe --build --assert -Wall -j $(JOBS) \
		--Mdir $(BUILD) --top-module bnn_sim_top \
		-CFLAGS "-std=c++17" \
		$(SIM_RTL) $(CORE_RTL) simple_test_main.cpp

# Build, generate a random case, run it, and verify the output image.
test: $(SIM)
	$(PYTHON) tests/gen_random_case.py $(SEED) $(TEST_DIR)
	$(SIM) $(TEST_DIR)/commands.hex $(TEST_DIR)/memory.img
	$(PYTHON) tests/check_random_case.py $(SEED) $(TEST_DIR)

# Run the committed seed-1 case.
run: $(SIM)
	$(SIM) $(DATA_DIR)/commands.hex $(DATA_DIR)/memory.img

# Regenerate the committed case; SEED defaults to 1.
gen:
	$(PYTHON) tests/gen_random_case.py $(SEED) $(DATA_DIR)

# Verify the last run of the committed case.
check:
	$(PYTHON) tests/check_random_case.py $(SEED) $(DATA_DIR)

clean:
	rm -rf $(BUILD)
	rm -f $(DATA_DIR)/memory_out.img

help:
	@printf '%s\n' \
		'make / make build   Build $(SIM)' \
		'make test [SEED=N]  Generate, run, and verify a random case' \
		'make run            Run the committed tests/case case' \
		'make gen [SEED=N]   Regenerate tests/case data' \
		'make check [SEED=N] Verify tests/case/memory_out.img' \
		'make clean          Remove build outputs'

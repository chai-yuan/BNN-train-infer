# Verilator + DPI-C regression flow for the BNN v2 simulation top.
VERILATOR ?= verilator
PYTHON    ?= python3
JOBS      ?= 2
MODEL     ?= ../../qwen3-zh-5k.gguf

BUILD     := build
DATA_DIR  := tests/case
SIM       := $(BUILD)/Vbnn_sim_top
PROFILE   := --trace $(BUILD)/case_trace.json --folded $(BUILD)/case_flame.folded \
             --report $(DATA_DIR)/memory_profile.txt

FP_PACKAGE := ../bnn_core/bnn_fp_math.sv
CORE_RTL   := $(FP_PACKAGE) $(filter-out $(FP_PACKAGE),$(wildcard ../bnn_core/*.sv))
SIM_RTL    := $(sort $(wildcard sim_rtl/*.sv))

.PHONY: all build test run gen check clean help
.DEFAULT_GOAL := all

all: build
build: $(SIM)

$(SIM): $(SIM_RTL) $(CORE_RTL) simple_test_main.cpp Makefile
	$(VERILATOR) --cc --exe --build --assert -Wall -j $(JOBS) \
		--Mdir $(BUILD) --top-module bnn_sim_top \
		-CFLAGS "-std=c++17" \
		$(SIM_RTL) $(CORE_RTL) simple_test_main.cpp

# Run the committed Qwen3 W1A8 case and verify it within the recorded tolerance.
test: $(SIM)
	$(SIM) $(DATA_DIR)/commands.hex $(DATA_DIR)/memory.img $(PROFILE)
	$(PYTHON) tests/check_model_case.py $(DATA_DIR)

# Run the committed case without checking.
run: $(SIM)
	$(SIM) $(DATA_DIR)/commands.hex $(DATA_DIR)/memory.img $(PROFILE)

# Regenerate tests/case from the Qwen3 GGUF (requires numpy and the model).
gen:
	$(PYTHON) tests/gen_model_case.py --model $(MODEL) --out $(DATA_DIR)

# Verify the last run of the committed case.
check:
	$(PYTHON) tests/check_model_case.py $(DATA_DIR)

clean:
	rm -rf $(BUILD)
	rm -f $(DATA_DIR)/memory_out.img $(DATA_DIR)/memory_profile.txt \
		$(DATA_DIR)/memory_trace.json $(DATA_DIR)/memory_flame.folded

help:
	@printf '%s\n' \
		'make / make build   Build $(SIM)' \
		'make test           Run the committed case and verify with golden tolerance' \
		'make run            Run the committed tests/case case' \
		'make gen [MODEL=K]  Regenerate tests/case from a Qwen3-zh-5k GGUF' \
		'make check          Verify tests/case/memory_out.img from the last run' \
		'make clean          Remove build outputs and generated profiles'

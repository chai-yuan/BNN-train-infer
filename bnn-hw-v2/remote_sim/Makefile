# Verilator + TCP co-simulation for the BNN v2 core.
VERILATOR ?= verilator
PYTHON    ?= python3
JOBS      ?= 2

BUILD     := build
SIM       := $(BUILD)/Vbnn_sim_top
CASE      := ../simple_test/tests/case
REFERENCE := $(CASE)/memory_out.img

FP_PACKAGE := ../bnn_core/bnn_fp_math.sv
CORE_RTL   := $(FP_PACKAGE) $(filter-out $(FP_PACKAGE),$(wildcard ../bnn_core/*.sv))
SIM_RTL    := $(sort $(wildcard ../bnn_sim_top/*.sv))

.PHONY: all build run test clean help
.DEFAULT_GOAL := all

all: build
build: $(SIM)

$(SIM): $(SIM_RTL) $(CORE_RTL) remote_sim_main.cpp Makefile
	$(VERILATOR) --cc --exe --build --assert -Wall -j $(JOBS) \
		--Mdir $(BUILD) --top-module bnn_sim_top \
		-CFLAGS "-std=c++17 -pthread" \
		$(SIM_RTL) $(CORE_RTL) remote_sim_main.cpp

run: $(SIM)
	$(SIM) --port 12345

# Run the committed case through the mock TCP client and compare the resulting
# memory image against the file-driven simple_test run.
test: $(SIM)
	@if [ ! -f $(REFERENCE) ]; then $(MAKE) -C ../simple_test test; fi
	$(PYTHON) tests/mock_client.py --server $(SIM) \
		--commands $(CASE)/commands.hex --memory $(CASE)/memory.img \
		--output $(BUILD)/memory_out.img --reference $(REFERENCE)

clean:
	rm -rf $(BUILD)

help:
	@printf '%s\n' \
		'make / make build   Build $(SIM)' \
		'make run            Start the TCP server on 127.0.0.1:12345' \
		'make test           Run the committed case through the mock TCP client' \
		'make clean          Remove build outputs'

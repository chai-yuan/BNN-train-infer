# Post-synthesis timing and area analysis for the BNN v2 core.
#
# Tool chain:
#   sv2v    SystemVerilog -> Verilog-2005 (Yosys cannot parse the SV package)
#   yosys   synthesis + ABC mapping with ICS55 standard cells
#   sta     OpenSTA static timing analysis
#   python3 report generation
#
# Usage:
#   make                      # synth + all corners + Markdown report
#   make synth                # sv2v + Yosys mapping + netlist + area log
#   make sta                  # OpenSTA for $(CORNERS)
#   make summarize            # build reports/<run>/report.md
#   make clean                # remove build/
#   make distclean            # remove build/ and reports/
#
# Typical overrides:
#   make FAMILY=H7CL
#   make CORNERS="ss_rcworst_1p08_125 ss_cworst_1p08_m40"
#   make CLK_PERIOD=80 CLK_UNC=0.2 TOP_N=10
#   make RUN=my-run
#
# Reports are written to reports/<RUN>/; generated netlists/logs go to build/.

SHELL := /bin/bash

PDK        ?= /home/charain/project/icsprout55-pdk
TOP        ?= bnn_core_top
FAMILY     ?= H7CR
MAP_CORNER ?= ss_rcworst_1p08_125
CORNERS    ?= typ_tt_1p2_25 ss_cworst_1p08_m40 ss_rcworst_1p08_125 ss_rcworst_1p2_m40
CLK_PORT   ?= clock
CLK_PERIOD ?= 100.0
CLK_UNC    ?= 0.1
TOP_N      ?= 10
IO_DELAY   ?=
OUT_LOAD   ?=
RUN        ?= $(shell date +%Y%m%d-%H%M%S)

YOSYS  ?= yosys
SV2V   ?= sv2v
STA    ?= sta
PYTHON ?= python3

BUILD      := build
REPORT_DIR := reports/$(RUN)

STD_ROOT   := $(PDK)/IP/STD_cell/ics55_LLSC_H7C_V1p10C100
LIB_PREFIX := ics55_LLSC_$(FAMILY)
STD_DIR    := $(STD_ROOT)/$(LIB_PREFIX)
MAP_LIB    := $(STD_DIR)/liberty/$(LIB_PREFIX)_$(MAP_CORNER)_nldm.lib
FAM_SUFFIX := $(patsubst H7C%,%,$(FAMILY))
BUF_CELL   := BUFX2H7$(FAM_SUFFIX)

CORE_SV    := $(sort $(wildcard ../bnn_core/*.sv))
SV2V_DEFS  ?= -D SYNTHESIS -D TARGET_ASIC
CONV_V     := $(BUILD)/$(TOP)_conv.v
RAW_NET    := $(BUILD)/$(TOP)_raw.v
NETLIST    := $(BUILD)/$(TOP)_sta.v
SYNTH_LOG  := $(BUILD)/synth.log
ABC_CONSTR := $(BUILD)/abc.constr
CONFIG     := $(BUILD)/config.mk

.PHONY: all report synth sta summarize tools-check clean distclean help print-config force

# report builds STA first and then summarizes; keep prerequisite order stable.
.NOTPARALLEL: report

all: report

force:

$(BUILD):
	@mkdir -p $@

$(ABC_CONSTR): Makefile | $(BUILD)
	@printf 'set_driving_cell %s\nset_load 0.01\n' '$(BUF_CELL)' > $@

# Rebuild the netlist only when the source, Makefile or synthesis configuration
# changes. STA-only iterations therefore reuse the existing netlist.
$(CONFIG): force | $(BUILD)
	@printf '%s\n' \
		'TOP=$(TOP)' \
		'FAMILY=$(FAMILY)' \
		'MAP_CORNER=$(MAP_CORNER)' \
		'PDK=$(PDK)' \
		'SV2V_DEFS=$(SV2V_DEFS)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv $@.tmp $@; fi

tools-check:
	@for t in '$(YOSYS)' '$(SV2V)' '$(STA)' '$(PYTHON)'; do \
		command -v $$t >/dev/null || { echo "ERROR: missing tool: $$t" >&2; exit 1; }; \
	done
	@test -f '$(MAP_LIB)' || { \
		echo "ERROR: missing liberty file: $(MAP_LIB)" >&2; \
		echo "       check PDK=..., FAMILY=..., MAP_CORNER=..." >&2; \
		exit 1; \
	}
	@test -d '$(STD_DIR)' || { echo "ERROR: missing library dir: $(STD_DIR)" >&2; exit 1; }

synth: $(NETLIST)

$(NETLIST): $(CORE_SV) Makefile $(CONFIG) $(ABC_CONSTR) | tools-check $(BUILD)
	@mkdir -p $(BUILD)
	$(SV2V) $(SV2V_DEFS) $(CORE_SV) > $(CONV_V).tmp
	@mv $(CONV_V).tmp $(CONV_V)
	@set -o pipefail; $(YOSYS) -p "\
		read_verilog -sv $(CONV_V); \
		hierarchy -check -top $(TOP); \
		synth -top $(TOP) -flatten; \
		dfflibmap -liberty $(MAP_LIB); \
		abc -liberty $(MAP_LIB) -constr $(ABC_CONSTR); \
		opt_clean; \
		stat -liberty $(MAP_LIB); \
		rename -unescape; \
		write_verilog -noattr -simple-lhs $(RAW_NET)" 2>&1 | tee $(SYNTH_LOG)
	@sed 's/wire signed /wire /g' $(RAW_NET) > $(NETLIST)
	@echo "netlist: $(NETLIST)"
	@echo "synth log: $(SYNTH_LOG)"

sta: $(NETLIST)
	@mkdir -p $(REPORT_DIR)
	@set -e; for c in $(CORNERS); do \
		lib="$(STD_DIR)/liberty/$(LIB_PREFIX)_$${c}_nldm.lib"; \
		test -f "$$lib" || { echo "ERROR: missing liberty file: $$lib" >&2; exit 1; }; \
		echo "=== STA $$c ==="; \
		TOP="$(TOP)" NETLIST="$(NETLIST)" LIB="$$lib" CLK_PORT="$(CLK_PORT)" \
		CLK_PERIOD="$(CLK_PERIOD)" CLK_UNC="$(CLK_UNC)" TOP_N="$(TOP_N)" \
		IO_DELAY="$(IO_DELAY)" OUT_LOAD="$(OUT_LOAD)" \
		JSON_OUT="$(REPORT_DIR)/sta_$${c}.json" \
		RPT_OUT="$(REPORT_DIR)/sta_$${c}.rpt" \
		$(STA) -no_init -no_splash -exit sta.tcl \
			> "$(REPORT_DIR)/sta_$${c}.log" 2>&1; \
	done
	@echo "raw STA reports: $(REPORT_DIR)"

summarize:
	@test -d '$(REPORT_DIR)' || { \
		echo "ERROR: missing report directory: $(REPORT_DIR)" >&2; \
		echo "       run 'make report' first, or pass RUN=<run> / RUN=latest" >&2; \
		exit 1; \
	}
	$(PYTHON) summarize.py \
		--run-dir '$(REPORT_DIR)' \
		--top '$(TOP)' \
		--family '$(FAMILY)' \
		--map-corner '$(MAP_CORNER)' \
		--corners '$(CORNERS)' \
		--period '$(CLK_PERIOD)' \
		--uncertainty '$(CLK_UNC)' \
		--io-delay '$(IO_DELAY)' \
		--out-load '$(OUT_LOAD)' \
		--top-n '$(TOP_N)' \
		--synth-log '$(SYNTH_LOG)' \
		--netlist '$(NETLIST)' \
		--liberty '$(MAP_LIB)'

report: sta summarize
	@if [ '$(RUN)' != 'latest' ]; then ln -sfn '$(RUN)' reports/latest; fi
	@echo "report: $(REPORT_DIR)/report.md"

clean:
	rm -rf $(BUILD)

distclean: clean
	rm -rf reports

print-config:
	@printf '%s\n' \
		'PDK=$(PDK)' \
		'TOP=$(TOP)' \
		'FAMILY=$(FAMILY)' \
		'MAP_CORNER=$(MAP_CORNER)' \
		'CORNERS=$(CORNERS)' \
		'CLK_PORT=$(CLK_PORT)' \
		'CLK_PERIOD=$(CLK_PERIOD)' \
		'CLK_UNC=$(CLK_UNC)' \
		'TOP_N=$(TOP_N)' \
		'MAP_LIB=$(MAP_LIB)' \
		'RUN=$(RUN)'

help:
	@printf '%s\n' \
		'make [report]          synth + STA + Markdown report' \
		'make synth             sv2v + Yosys synthesis/mapping/area' \
		'make sta               run OpenSTA over $(CORNERS)' \
		'make summarize         generate reports/<RUN>/report.md (use RUN=latest)' \
		'make print-config      show resolved paths and options' \
		'make clean             remove build/' \
		'make distclean         remove build/ and reports/' \
		'' \
		'Variables:' \
		'  PDK=$(PDK)' \
		'  FAMILY=H7CR|H7CL|H7CH (RVT/LVT/HVT)' \
		'  MAP_CORNER=<corner used for ABC mapping>' \
		'  CORNERS="<space separated Liberty corner suffixes>"' \
		'  CLK_PERIOD=<probe clock period in ns>' \
		'  CLK_UNC=<clock uncertainty in ns>' \
		'  TOP_N=<number of critical paths in the report>' \
		'  IO_DELAY=<optional input/output delay in ns>' \
		'  OUT_LOAD=<optional output load in pF>' \
		'  RUN=<report directory name>'

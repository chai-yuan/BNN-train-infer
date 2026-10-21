ACCE ?= soft

ifeq ($(filter $(ACCE),soft verilator),)
$(error Unsupported ACCE '$(ACCE)'; available accelerator backends: soft verilator)
endif

CC ?= cc
CXX ?= c++
AR ?= ar
PYTHON ?= python3
VERILATOR ?= verilator
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic
CXXFLAGS ?= -O2 -Wall -Wextra
# accelerator 的数值契约要求逐组 FP32 累加，不允许 fast-math 或 FMA 收缩。
# override 保证命令行 CFLAGS= 覆盖时契约 flag 仍被追加；sanitizer 等用环境变量传入。
override CFLAGS += -std=c99 -fno-fast-math -ffp-contract=off
override CXXFLAGS += -std=c++17 -fno-fast-math -ffp-contract=off
LDLIBS += -lm
LINKER := $(CC)
LINK_FLAGS := $(filter-out -std=c99,$(CFLAGS))

BUILD_DIR := build/$(ACCE)
ENGINE_SOURCES := bnn-sw/bnn-engine.c bnn-sw/common.c bnn-sw/runtime.c \
                  bnn-sw/model-mlp.c bnn-sw/model-cnn.c bnn-sw/model-gpt.c \
                  bnn-sw/model-llama.c
ACCEL_SOURCE := bnn-accelerator/$(ACCE)-bnn-accelerator.c
ENGINE_OBJECTS := $(addprefix $(BUILD_DIR)/,$(ENGINE_SOURCES:.c=.o) $(ACCEL_SOURCE:.c=.o))
MAIN_OBJECT := $(BUILD_DIR)/bnn-infer.o
LIBRARY := $(BUILD_DIR)/libbnn-engine.a
DEPS := $(ENGINE_OBJECTS:.o=.d) $(MAIN_OBJECT:.o=.d)

ifeq ($(ACCE),verilator)
LINKER := $(CXX)
LINK_FLAGS += $(filter-out -std=c++17,$(CXXFLAGS))
LDLIBS += -pthread -latomic
RTL_SOURCES := $(wildcard bnn-hw/bnn_core/*.sv)
RTL_DIR := $(abspath $(BUILD_DIR)/rtl)
RTL_LIBRARY := $(RTL_DIR)/Vbnn_core_top__ALL.a
VERILATOR_ROOT := $(shell $(VERILATOR) --getenv VERILATOR_ROOT)
RTL_INCLUDES := -I$(RTL_DIR) -isystem $(VERILATOR_ROOT)/include -isystem $(VERILATOR_ROOT)/include/vltstd
BRIDGE_OBJECT := $(BUILD_DIR)/bnn-accelerator/verilator-bnn-bridge.o
RTL_RUNTIME_OBJECTS := $(BUILD_DIR)/verilated.o $(BUILD_DIR)/verilated_threads.o \
                       $(BUILD_DIR)/verilated_vcd_c.o
BACKEND_OBJECTS := $(BRIDGE_OBJECT) $(RTL_RUNTIME_OBJECTS)
DEPS += $(BACKEND_OBJECTS:.o=.d)

$(RTL_LIBRARY): $(RTL_SOURCES) Makefile
	@mkdir -p $(RTL_DIR)
	$(VERILATOR) --cc --assert --trace -Wall --top-module bnn_core_top \
		--Mdir $(RTL_DIR) $(RTL_SOURCES)
	$(MAKE) -C $(RTL_DIR) -f Vbnn_core_top.mk CXX="$(CXX)" \
		CXXFLAGS="$(CXXFLAGS)" Vbnn_core_top__ALL.a
	@touch $@

$(BRIDGE_OBJECT): bnn-accelerator/verilator-bnn-bridge.cpp $(RTL_LIBRARY)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RTL_INCLUDES) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: $(VERILATOR_ROOT)/include/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RTL_INCLUDES) -pthread -MMD -MP -c $< -o $@
endif

.DEFAULT_GOAL := all
# 后端共享根目录入口，每次显式链接以免切换 ACCE 时复用上一个后端的可执行文件。
.PHONY: all bnn-infer clean test test-accelerator test-trace
all: bnn-infer $(LIBRARY)

bnn-infer: $(MAIN_OBJECT) $(LIBRARY)
	$(LINKER) $(LINK_FLAGS) $(LDFLAGS) -o $@ $(MAIN_OBJECT) $(LIBRARY) $(LDLIBS)

$(LIBRARY): $(ENGINE_OBJECTS) $(BACKEND_OBJECTS) $(RTL_LIBRARY) Makefile
ifeq ($(ACCE),verilator)
	@{ printf 'CREATE %s\n' '$@.tmp'; \
		for obj in $(ENGINE_OBJECTS) $(BACKEND_OBJECTS); do printf 'ADDMOD %s\n' "$$obj"; done; \
		printf 'ADDLIB %s\nSAVE\nEND\n' '$(RTL_LIBRARY)'; \
	} | $(AR) -M
	mv $@.tmp $@
else
	$(AR) rcs $@ $(ENGINE_OBJECTS)
endif

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

test: all
	$(PYTHON) tests/test_mlp.py --binary ./bnn-infer --library $(LIBRARY) \
		--cc "$(CC)" --linker "$(LINKER)" --ldlibs="$(LDLIBS)"
	@if $(PYTHON) -c "import numpy" 2>/dev/null; then \
		$(PYTHON) tests/test_llama.py; \
	elif [ -x "./.venv/bin/python3" ]; then \
		./.venv/bin/python3 tests/test_llama.py; \
	fi

ifeq ($(ACCE),verilator)
test-accelerator: $(LIBRARY)
	$(PYTHON) tests/test_accelerator.py --library $(LIBRARY) --cc "$(CC)" --cxx "$(CXX)" --verilator "$(VERILATOR)"
test-trace: $(LIBRARY)
	$(PYTHON) tests/test_verilator_trace.py --library $(LIBRARY) --cc "$(CC)" --cxx "$(CXX)"
else
test-accelerator test-trace:
	@printf 'Use ACCE=verilator for backend differential and waveform tests.\n'
	@exit 1
endif

clean:
	$(RM) bnn-infer $(MAIN_OBJECT) $(ENGINE_OBJECTS) $(DEPS) $(LIBRARY) $(BACKEND_OBJECTS)
ifeq ($(ACCE),verilator)
	$(RM) -r $(RTL_DIR)
endif

-include $(DEPS)

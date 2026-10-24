# =============================================================================
# BNN inference build
#
# Backend selection:
#   make ACCE=soft       C software accelerator backend (default)
#   make ACCE=verilator  Verilator RTL simulation backend
#
# Test program builds and runs are intentionally not part of this Makefile.
# =============================================================================

ACCE ?= soft

ifeq ($(filter $(ACCE),soft verilator),)
$(error Unsupported ACCE '$(ACCE)'; available accelerator backends: soft verilator)
endif

.DEFAULT_GOAL := all

# =============================================================================
# Toolchain and compiler flags
# =============================================================================

CC ?= cc
CXX ?= c++
AR ?= ar
VERILATOR ?= verilator

CFLAGS ?= -O2 -Wall -Wextra -Wpedantic
CXXFLAGS ?= -O2 -Wall -Wextra

# The accelerator numeric contract requires per-group FP32 accumulation.
# These flags are appended even when CFLAGS/CXXFLAGS are overridden.
override CFLAGS += -std=c99 -fno-fast-math -ffp-contract=off
override CXXFLAGS += -std=c++17 -fno-fast-math -ffp-contract=off

LDLIBS += -lm
LINKER := $(CC)
LINK_FLAGS := $(filter-out -std=c99,$(CFLAGS))

# =============================================================================
# Common source and object layout
# =============================================================================

BUILD_DIR := build/$(ACCE)

ENGINE_SOURCES := bnn-sw/bnn-engine.c bnn-sw/common.c bnn-sw/runtime.c \
                  bnn-sw/model-mlp.c bnn-sw/model-cnn.c bnn-sw/model-gpt.c \
                  bnn-sw/model-llama.c bnn-sw/gguf.c bnn-sw/tokenizer-bpe.c \
                  bnn-sw/model-qwen3.c
ACCEL_SOURCE := bnn-accelerator/$(ACCE)-bnn-accelerator.c

ENGINE_OBJECTS := $(addprefix $(BUILD_DIR)/,$(ENGINE_SOURCES:.c=.o) $(ACCEL_SOURCE:.c=.o))
MAIN_OBJECT := $(BUILD_DIR)/bnn-infer.o
LIBRARY := $(BUILD_DIR)/libbnn-engine.a

# Backend-specific inputs. Both stay empty for the soft backend.
BACKEND_OBJECTS :=
RTL_LIBRARY :=

DEPS := $(ENGINE_OBJECTS:.o=.d) $(MAIN_OBJECT:.o=.d)

# =============================================================================
# Verilator backend
# =============================================================================

ifeq ($(ACCE),verilator)

LINKER := $(CXX)
LINK_FLAGS += $(filter-out -std=c++17,$(CXXFLAGS))
LDLIBS += -pthread -latomic

RTL_SOURCES := $(wildcard bnn-hw/bnn_core/*.sv)
RTL_DIR := $(abspath $(BUILD_DIR)/rtl)
RTL_LIBRARY := $(RTL_DIR)/Vbnn_core_top__ALL.a

VERILATOR_ROOT := $(shell $(VERILATOR) --getenv VERILATOR_ROOT)
RTL_INCLUDES := -I$(RTL_DIR) \
                -isystem $(VERILATOR_ROOT)/include \
                -isystem $(VERILATOR_ROOT)/include/vltstd

BRIDGE_OBJECT := $(BUILD_DIR)/bnn-accelerator/verilator-bnn-bridge.o
RTL_RUNTIME_OBJECTS := $(BUILD_DIR)/verilated.o \
                       $(BUILD_DIR)/verilated_threads.o \
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

# =============================================================================
# Build rules
# =============================================================================

# Both backends write the same root entry point, so force a fresh link when
# switching ACCE instead of reusing the previous backend's executable.
.PHONY: all bnn-infer clean

all: bnn-infer

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(LIBRARY): $(ENGINE_OBJECTS) $(BACKEND_OBJECTS) $(RTL_LIBRARY) Makefile
ifeq ($(ACCE),verilator)
# The Verilator model is a nested archive; flatten it into the final archive
# so embedding programs only need to link one library.
	@{ printf 'CREATE %s\n' '$@.tmp'; \
	   for obj in $(ENGINE_OBJECTS) $(BACKEND_OBJECTS); do \
	       printf 'ADDMOD %s\n' "$$obj"; done; \
	   printf 'ADDLIB %s\nSAVE\nEND\n' '$(RTL_LIBRARY)'; \
	} | $(AR) -M
	mv $@.tmp $@
else
	$(AR) rcs $@ $(ENGINE_OBJECTS)
endif

bnn-infer: $(MAIN_OBJECT) $(LIBRARY)
	$(LINKER) $(LINK_FLAGS) $(LDFLAGS) -o $@ $(MAIN_OBJECT) $(LIBRARY) $(LDLIBS)

clean:
	$(RM) bnn-infer $(MAIN_OBJECT) $(ENGINE_OBJECTS) $(DEPS) $(LIBRARY) $(BACKEND_OBJECTS)
ifeq ($(ACCE),verilator)
	$(RM) -r $(RTL_DIR)
endif

-include $(DEPS)

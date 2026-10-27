# =============================================================================
# BNN v2 inference build
#
# Only the C software reference backend is supported:
#   make ACCE=soft       (default)
# RTL self-tests are optional and built independently in bnn-hw/bnn_core_test.
#
# Only bnn-sw-v2 + bnn-accelerator are built. The legacy bnn-sw / bnn-infer
# implementation was removed; it remains available in git history.
# Test program builds and runs are intentionally not part of this Makefile.
# =============================================================================

ACCE ?= soft

ifneq ($(ACCE),soft)
$(error Unsupported ACCE '$(ACCE)'; only ACCE=soft is supported)
endif

.DEFAULT_GOAL := all

# =============================================================================
# Toolchain and compiler flags
# =============================================================================

CC ?= cc
AR ?= ar

CFLAGS ?= -O2 -Wall -Wextra -Wpedantic

# The accelerator numeric contract requires per-group FP32 accumulation.
# These flags are appended even when CFLAGS are overridden.
override CFLAGS += -std=c11 -fno-fast-math -ffp-contract=off

LDLIBS += -lm
LINK_FLAGS := $(filter-out -std=c11,$(CFLAGS))

# =============================================================================
# Common source and object layout
# =============================================================================

BUILD_DIR := build/soft

ENGINE_SOURCES := bnn-sw-v2/common.c bnn-sw-v2/engine.c bnn-sw-v2/linear.c \
                  bnn-sw-v2/gguf.c bnn-sw-v2/tokenizer/tokenizer-bpe.c bnn-sw-v2/registry.c \
                  bnn-sw-v2/bnn-mlp/bnn-mlp.c bnn-sw-v2/qwen3/qwen3.c
ACCEL_SOURCE := bnn-accelerator/soft-bnn-accelerator.c

ENGINE_OBJECTS := $(addprefix $(BUILD_DIR)/,$(ENGINE_SOURCES:.c=.o) $(ACCEL_SOURCE:.c=.o))
MAIN_OBJECT := $(BUILD_DIR)/bnn-infer-v2.o
LIBRARY := $(BUILD_DIR)/libbnn-engine.a

DEPS := $(ENGINE_OBJECTS:.o=.d) $(MAIN_OBJECT:.o=.d)

# =============================================================================
# Build rules
# =============================================================================

# Always relink the CLI so an existing executable from the removed backend
# cannot be mistaken for a soft build.
.PHONY: all bnn-infer-v2 clean

all: bnn-infer-v2

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(LIBRARY): $(ENGINE_OBJECTS) Makefile
	$(AR) rcs $@ $(ENGINE_OBJECTS)

bnn-infer-v2: $(MAIN_OBJECT) $(LIBRARY)
	$(CC) $(LINK_FLAGS) $(LDFLAGS) -o $@ $(MAIN_OBJECT) $(LIBRARY) $(LDLIBS)

clean:
	$(RM) bnn-infer-v2
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

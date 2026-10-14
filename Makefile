ACCE ?= soft

ifneq ($(ACCE),soft)
$(error Unsupported ACCE '$(ACCE)'; available accelerator backends: soft)
endif

CC ?= cc
AR ?= ar
PYTHON ?= python3
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic
# accelerator 的数值契约要求逐组 FP32 累加，不允许 fast-math 或 FMA 收缩。
# override 保证命令行 CFLAGS= 覆盖时契约 flag 仍被追加；sanitizer 等用环境变量传入。
override CFLAGS += -std=c99 -fno-fast-math -ffp-contract=off
LDLIBS += -lm

BUILD_DIR := build/$(ACCE)
ENGINE_SOURCES := bnn-sw/bnn-engine.c bnn-sw/common.c bnn-sw/runtime.c \
                  bnn-sw/model-mlp.c bnn-sw/model-cnn.c bnn-sw/model-gpt.c \
                  bnn-sw/model-llama.c
ACCEL_SOURCE := bnn-accelerator/$(ACCE)-bnn-accelerator.c
ENGINE_OBJECTS := $(addprefix $(BUILD_DIR)/,$(ENGINE_SOURCES:.c=.o) $(ACCEL_SOURCE:.c=.o))
MAIN_OBJECT := $(BUILD_DIR)/bnn-infer.o
LIBRARY := $(BUILD_DIR)/libbnn-engine.a
DEPS := $(ENGINE_OBJECTS:.o=.d) $(MAIN_OBJECT:.o=.d)

.PHONY: all clean test
all: bnn-infer $(LIBRARY)

bnn-infer: $(MAIN_OBJECT) $(LIBRARY)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(MAIN_OBJECT) $(LIBRARY) $(LDLIBS)

$(LIBRARY): $(ENGINE_OBJECTS)
	$(AR) rcs $@ $^

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

test: all
	$(PYTHON) tests/test_mlp.py --binary ./bnn-infer --library $(LIBRARY)
	@if $(PYTHON) -c "import numpy" 2>/dev/null; then \
		$(PYTHON) tests/test_llama.py; \
	elif [ -x "./.venv/bin/python3" ]; then \
		./.venv/bin/python3 tests/test_llama.py; \
	fi

clean:
	$(RM) bnn-infer $(MAIN_OBJECT) $(ENGINE_OBJECTS) $(DEPS) $(LIBRARY)

-include $(DEPS)

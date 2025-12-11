# Makefile for HipKittens FP8 GEMM
# 
# Prerequisites:
#   - ROCm 6.0+
#   - HipKittens (https://github.com/HazyResearch/HipKittens)
#
# Usage:
#   make                    # Build for auto-detected GPU
#   make ARCH=gfx942        # Build for MI300X
#   make ARCH=gfx950        # Build for MI355X
#   make USE_HIPKITTENS=1   # Build with HipKittens (requires HK installed)

# Compiler
HIPCC ?= /opt/rocm/bin/hipcc

# Auto-detect GPU architecture if not specified
ifndef ARCH
    ARCH := $(shell rocminfo 2>/dev/null | grep -o 'gfx9[0-9]*' | head -1)
endif
ifeq ($(ARCH),)
    ARCH := gfx942
endif

# HipKittens path (set HIPKITTENS_ROOT environment variable)
HIPKITTENS_ROOT ?= $(HOME)/HipKittens

# Compiler flags
CXXFLAGS := -O3 -std=c++17 --offload-arch=$(ARCH)
CXXFLAGS += -ffast-math -funsafe-math-optimizations

# HipKittens support
ifdef USE_HIPKITTENS
    CXXFLAGS += -DUSE_HIPKITTENS -I$(HIPKITTENS_ROOT)/include
endif

# Source files
SRCS := fp8_gemm_decode.hip
OBJS := $(SRCS:.hip=.o)
TARGET := libhk_fp8_gemm.so

# Targets
.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(HIPCC) -shared -o $@ $^ $(CXXFLAGS)

%.o: %.hip
	$(HIPCC) -c -fPIC -o $@ $< $(CXXFLAGS)

# Generate code object for AITER integration
fp8_gemm_decode_$(ARCH).co: fp8_gemm_decode.hip
	$(HIPCC) --genco -o $@ $< $(CXXFLAGS)

co: fp8_gemm_decode_$(ARCH).co

clean:
	rm -f $(OBJS) $(TARGET) *.co

test: $(TARGET)
	python3 test_python.py

# Print configuration
info:
	@echo "Architecture: $(ARCH)"
	@echo "HipKittens root: $(HIPKITTENS_ROOT)"
	@echo "Compiler: $(HIPCC)"
	@echo "Flags: $(CXXFLAGS)"




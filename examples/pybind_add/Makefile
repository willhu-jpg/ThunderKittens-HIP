# Compiler
GPU=MI300

TARGET=simple_tk
SRC=example_add.hip


# ROCM 
COMMON_INCLUDE_DIR := ../../Common
GPU_RUNTIME := HIP

# HIP variables
ROCM_INSTALL_DIR := /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Common variables and flags
CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I $(COMMON_INCLUDE_DIR) -I${THUNDERKITTENS_ROOT}/include -I/opt/rocm/include/hip
ILDFLAGS  :=
ILDLIBS   :=
ARCH	  := --offload-arch=gfx942 # MI300X

CXXFLAGS ?= -Wall -Wextra

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)


ICXXFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC -lpython3.10


# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $(SRC) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) $(ARCH) -o $(TARGET)$(shell python3-config --extension-suffix)

# Clean target
clean:
	rm -f $(TARGET)
# ------------------------------------------------------------------------
# Root-level Makefile for CUDA modules: reduce, axpy, dot, add
# ------------------------------------------------------------------------

# Compiler and flags
NVCC        := nvcc
NVCCFLAGS   := -I include -arch=sm_90

# cuBLAS linking flags (only for axpy and dot)
CUBLAS_LIBS := -lcublas

# Directories
SRC_DIR     := src
BUILD_DIR   := build

# Source files
REDUCE_SRC  := $(SRC_DIR)/reduce/main.cu
AXPY_SRC    := $(SRC_DIR)/axpy/main.cu
DOT_SRC     := $(SRC_DIR)/dot/main.cu
ADD_SRC     := $(SRC_DIR)/add/main.cu

# Output binaries
REDUCE_BIN  := $(BUILD_DIR)/reduction
AXPY_BIN    := $(BUILD_DIR)/axpy
DOT_BIN     := $(BUILD_DIR)/dot
ADD_BIN     := $(BUILD_DIR)/add

# ------------------------------------------------------------------------
# Default target
# ------------------------------------------------------------------------
.PHONY: all
all: build

# ------------------------------------------------------------------------
# Build each module
# ------------------------------------------------------------------------
.PHONY: build
build: $(REDUCE_BIN) $(AXPY_BIN) $(DOT_BIN) $(ADD_BIN)

# Ensure build directory exists
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Build reduction (no cuBLAS)
$(REDUCE_BIN): $(REDUCE_SRC) | $(BUILD_DIR)
	@echo "Building reduction module..."
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Build axpy (link against cuBLAS)
$(AXPY_BIN): $(AXPY_SRC) | $(BUILD_DIR)
	@echo "Building axpy module..."
	$(NVCC) $(NVCCFLAGS) $(CUBLAS_LIBS) -o $@ $<

# Build dot product (link against cuBLAS)
$(DOT_BIN): $(DOT_SRC) | $(BUILD_DIR)
	@echo "Building dot product module..."
	$(NVCC) $(NVCCFLAGS) $(CUBLAS_LIBS) -o $@ $<

# Build add (no cuBLAS)
$(ADD_BIN): $(ADD_SRC) | $(BUILD_DIR)
	@echo "Building add module..."
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# ------------------------------------------------------------------------
# Run targets (all “run-<module>” depend on the full build first)
# ------------------------------------------------------------------------
.PHONY: run
run: run-reduce run-axpy run-dot run-add

.PHONY: run-reduce
run-reduce: build $(REDUCE_BIN)
	@echo "Running reduction module..."
	@$(REDUCE_BIN)

.PHONY: run-axpy
run-axpy: build $(AXPY_BIN)
	@echo "Running axpy module..."
	@$(AXPY_BIN)

.PHONY: run-dot
run-dot: build $(DOT_BIN)
	@echo "Running dot product module..."
	@$(DOT_BIN)

.PHONY: run-add
run-add: build $(ADD_BIN)
	@echo "Running add module..."
	@$(ADD_BIN)

# For convenience, you can also invoke “make reduce” instead of “make run-reduce”
# Each of these will build everything first, then run just that one binary.
.PHONY: reduce
reduce: run-reduce

.PHONY: axpy
axpy: run-axpy

.PHONY: dot
dot: run-dot

.PHONY: add
add: run-add

# ------------------------------------------------------------------------
# Clean up
# ------------------------------------------------------------------------
.PHONY: clean
clean:
	@echo "Cleaning up build files..."
	@rm -rf $(BUILD_DIR)

# ------------------------------------------------------------------------
# End of Makefile
# ------------------------------------------------------------------------

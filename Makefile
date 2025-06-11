# ------------------------------------------------------------------------
# Root‐level Makefile for CUDA modules: reduce, axpy, dot, add
# ------------------------------------------------------------------------

# Compiler and flags
NVCC        := nvcc
NVCCFLAGS   := -I include -arch=sm_90

# cuBLAS linking flags (only for axpy and dot)
CUBLAS_LIBS := -lcublas

# NCCL linking flags (only for allgather and allreduce)
NCCL_LIBS   := -lnccl

# Directories
SRC_DIR     := src
BUILD_DIR   := build

# Source files
REDUCE_SRC  := $(SRC_DIR)/reduce/main.cu
AXPY_SRC    := $(SRC_DIR)/axpy/main.cu
DOT_SRC     := $(SRC_DIR)/dot/main.cu
ADD_SRC     := $(SRC_DIR)/add/main.cu
GEMM_SRC    := $(SRC_DIR)/gemm/main.cu
BROADCAST_SRC := $(SRC_DIR)/collectives/broadcast.cu
ALLGATHER_SRC := $(SRC_DIR)/collectives/allgather.cu
ALLREDUCE_SRC := $(SRC_DIR)/collectives/allreduce.cu

# Output binaries
REDUCE_BIN  := $(BUILD_DIR)/reduction
AXPY_BIN    := $(BUILD_DIR)/axpy
DOT_BIN     := $(BUILD_DIR)/dot
ADD_BIN     := $(BUILD_DIR)/add
GEMM_BIN    := $(BUILD_DIR)/gemm
BROADCAST_BIN := $(BUILD_DIR)/broadcast
ALLGATHER_BIN := $(BUILD_DIR)/allgather
ALLREDUCE_BIN := $(BUILD_DIR)/allreduce

# ------------------------------------------------------------------------
# 1) Top‐level “build” target (for CI)
# ------------------------------------------------------------------------
.PHONY: build
build: $(REDUCE_BIN) $(AXPY_BIN) $(DOT_BIN) $(ADD_BIN) $(GEMM_BIN) $(BROADCAST_BIN) $(ALLGATHER_BIN) $(ALLREDUCE_BIN)
	@echo "All modules built. (No execution.)"

# ------------------------------------------------------------------------
# 2) Per‐module build+run targets
#
# Running `make add` will:
#   1) Build `$(ADD_BIN)` if it’s out of date,
#   2) Then run `$(ADD_BIN)`.
# ------------------------------------------------------------------------
.PHONY: add
add: $(ADD_BIN)
	@echo "Running add module…"
	@$(ADD_BIN)

.PHONY: axpy
axpy: $(AXPY_BIN)
	@echo "Running axpy module…"
	@$(AXPY_BIN)

.PHONY: dot
dot: $(DOT_BIN)
	@echo "Running dot product module…"
	@$(DOT_BIN)

.PHONY: reduce
reduce: $(REDUCE_BIN)
	@echo "Running reduction module…"
	@$(REDUCE_BIN)

.PHONY: gemm
gemm: $(GEMM_BIN)
	@echo "Running GEMM module…"
	@$(GEMM_BIN)

.PHONY: broadcast
broadcast: $(BROADCAST_BIN)
	@echo "Running broadcast module…"
	@$(BROADCAST_BIN)

.PHONY: allgather
allgather: $(ALLGATHER_BIN)
	@echo "Running allgather module…"
	@$(ALLGATHER_BIN)

.PHONY: allreduce
allreduce: $(ALLREDUCE_BIN)
	@echo "Running allreduce module…"
	@$(ALLREDUCE_BIN)

# ------------------------------------------------------------------------
# 3) Rules to compile each binary
#
# Notice: instead of having a separate “mkdir $(BUILD_DIR)” target,
# we do `@mkdir -p $(BUILD_DIR)` inline in each rule. This removes
# any ambiguity between a phony “build” target and a real “build” folder.
# ------------------------------------------------------------------------
$(REDUCE_BIN): $(REDUCE_SRC)
	@echo "Building reduction module…"
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(AXPY_BIN): $(AXPY_SRC)
	@echo "Building axpy module…"
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUBLAS_LIBS) -o $@ $<

$(DOT_BIN): $(DOT_SRC)
	@echo "Building dot product module…"
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUBLAS_LIBS) -o $@ $<

$(ADD_BIN): $(ADD_SRC)
	@echo "Building add module…"
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(GEMM_BIN): $(GEMM_SRC)
	@echo "Building GEMM module…"
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUBLAS_LIBS) -o $@ $<

$(BROADCAST_BIN): $(BROADCAST_SRC)
	@echo "Building broadcast module…"
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(ALLGATHER_BIN): $(ALLGATHER_SRC)
	@echo "Building allgather module…"
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(ALLREDUCE_BIN): $(ALLREDUCE_SRC)
	@echo "Building allreduce module…"
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# ------------------------------------------------------------------------
# 4) Clean everything
# ------------------------------------------------------------------------
.PHONY: clean
clean:
	@echo "Cleaning up build files…"
	@rm -rf $(BUILD_DIR)

# ------------------------------------------------------------------------
# End of Makefile
# ------------------------------------------------------------------------

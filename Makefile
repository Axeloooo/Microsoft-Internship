# ------------------------------------------------------------------------
# Root‐level Makefile for CUDA modules: reduce, axpy, dot, add
# ------------------------------------------------------------------------

# Compiler and flags
NVCC        := nvcc	
NVCCFLAGS   := -I include -arch=sm_90
MPICXX      := mpic++
MPIRUN      := mpirun
CUBLAS_LIBS := -lcublas
NCCL_LIBS   := -lnccl

# Directories
SRC_DIR     := src
BUILD_DIR   := build

# Kernels source files
REDUCE_SRC  := $(SRC_DIR)/kernels/reduce/main.cu
AXPY_SRC    := $(SRC_DIR)/kernels/axpy/main.cu
DOT_SRC     := $(SRC_DIR)/kernels/dot/main.cu
ADD_SRC     := $(SRC_DIR)/kernels/add/main.cu
GEMM_SRC    := $(SRC_DIR)/kernels/gemm/main.cu

# MCCL Collective source files
MCCL_SRC := $(SRC_DIR)/collectives/mccl/mccl.cu
MCCL_BROADCAST_SRC := $(SRC_DIR)/collectives/mccl/broadcast.cu
MCCL_ALLGATHER_SRC := $(SRC_DIR)/collectives/mccl/allgather.cu
MCCL_ALLREDUCE_SRC := $(SRC_DIR)/collectives/mccl/allreduce.cu
MCCL_REDUCE_SRC := $(SRC_DIR)/collectives/mccl/reduce.cu
MCCL_REDUCESCATTER_SRC := $(SRC_DIR)/collectives/mccl/reducescatter.cu

# MPI Collective source files
MPI_BROADCAST_SRC := $(SRC_DIR)/collectives/mpi/broadcast.cu
MPI_ALLGATHER_SRC := $(SRC_DIR)/collectives/mpi/allgather.cu
MPI_ALLREDUCE_SRC := $(SRC_DIR)/collectives/mpi/allreduce.cu
MPI_REDUCE_SRC := $(SRC_DIR)/collectives/mpi/reduce.cu
MPI_REDUCESCATTER_SRC := $(SRC_DIR)/collectives/mpi/reducescatter.cu

# Utility source files
TIMER_SRC   := $(SRC_DIR)/utils/timer.cu
DATA_SRC    := $(SRC_DIR)/utils/data.cu

# Kernels binaries
REDUCE_BIN  := $(BUILD_DIR)/kernels/reduction
AXPY_BIN    := $(BUILD_DIR)/kernels/axpy
DOT_BIN     := $(BUILD_DIR)/kernels/dot
ADD_BIN     := $(BUILD_DIR)/kernels/add
GEMM_BIN    := $(BUILD_DIR)/kernels/gemm

# MCCL Collective binaries
MCCL_BROADCAST_BIN := $(BUILD_DIR)/collectives/mccl/broadcast
MCCL_ALLGATHER_BIN := $(BUILD_DIR)/collectives/mccl/allgather
MCCL_ALLREDUCE_BIN := $(BUILD_DIR)/collectives/mccl/allreduce
MCCL_REDUCE_BIN := $(BUILD_DIR)/collectives/mccl/reduce
MCCL_REDUCESCATTER_BIN := $(BUILD_DIR)/collectives/mccl/reducescatter

# MPI Collective binaries
MPI_BROADCAST_BIN := $(BUILD_DIR)/collectives/mpi/broadcast
MPI_ALLGATHER_BIN := $(BUILD_DIR)/collectives/mpi/allgather
MPI_ALLREDUCE_BIN := $(BUILD_DIR)/collectives/mpi/allreduce
MPI_REDUCE_BIN := $(BUILD_DIR)/collectives/mpi/reduce
MPI_REDUCESCATTER_BIN := $(BUILD_DIR)/collectives/mpi/reducescatter

# ---------------------------------------------------------------------------
# Help target
# ---------------------------------------------------------------------------
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build           - Compile all modules (no execution)"
	@echo ""
	@echo "Kernels:"
	@echo "  add             - Build and run add kernel"
	@echo "  axpy            - Build and run axpy kernel"
	@echo "  dot             - Build and run dot product kernel"
	@echo "  reduce          - Build and run reduction kernel"
	@echo "  gemm            - Build and run GEMM kernel"
	@echo ""
	@echo "MCCL Collectives:"
	@echo "  mccl_broadcast  - Build and run MCCL broadcast"
	@echo "  mccl_allgather  - Build and run MCCL allgather"
	@echo "  mccl_allreduce  - Build and run MCCL allreduce"
	@echo "  mccl_reduce     - Build and run MCCL reduce"
	@echo "  mccl_reducescatter - Build and run MCCL reducescatter"
	@echo ""
	@echo "MPI Collectives:"
	@echo "  mpi_broadcast   - Build and run MPI broadcast (4 processes)"
	@echo "  mpi_allgather   - Build and run MPI allgather (4 processes)"
	@echo "  mpi_allreduce   - Build and run MPI allreduce (4 processes)"
	@echo "  mpi_reduce      - Build and run MPI reduce (4 processes)"
	@echo "  mpi_reducescatter - Build and run MPI reducescatter (4 processes)"
	@echo ""
	@echo "Utilities:"
	@echo "  clean           - Clean all build files"
	@echo "  help            - Show this help message"

# ---------------------------------------------------------------------------
# Compile all modules
# ---------------------------------------------------------------------------
.PHONY: build
build: $(REDUCE_BIN) $(AXPY_BIN) $(DOT_BIN) $(ADD_BIN) $(GEMM_BIN) $(MCCL_BROADCAST_BIN) $(MCCL_ALLGATHER_BIN) $(MCCL_ALLREDUCE_BIN) $(MCCL_REDUCE_BIN) $(MCCL_REDUCESCATTER_BIN) $(MPI_BROADCAST_BIN) $(MPI_ALLGATHER_BIN) $(MPI_ALLREDUCE_BIN) $(MPI_REDUCE_BIN) $(MPI_REDUCESCATTER_BIN)
	@echo "All modules built. (No execution.)"

# ------------------------------------------------------------------------
# Compile and run kernels
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

# ------------------------------------------------------------------------
# Compile and run MCCL collective modules
# ------------------------------------------------------------------------
.PHONY: mccl_broadcast
mccl_broadcast: $(MCCL_BROADCAST_BIN)
	@echo "Running MCCL broadcast module…"
	@echo ${MPIRUN} -np 8 $(MCCL_BROADCAST_BIN)
	@$(MPIRUN) -np 8 $(MCCL_BROADCAST_BIN)

.PHONY: mccl_allgather
mccl_allgather: $(MCCL_ALLGATHER_BIN)
	@echo "Running MCCL allgather module…"
	@echo ${MPIRUN} -np 8 $(MCCL_ALLGATHER_BIN)
	@$(MPIRUN) -np 8 $(MCCL_ALLGATHER_BIN)	

.PHONY: mccl_allreduce
mccl_allreduce: $(MCCL_ALLREDUCE_BIN)
	@echo "Running MCCL allreduce module…"
	@echo ${MPIRUN} -np 8 $(MCCL_ALLREDUCE_BIN)
	@$(MPIRUN) -np 8 $(MCCL_ALLREDUCE_BIN)	

.PHONY: mccl_reduce
mccl_reduce: $(MCCL_REDUCE_BIN)
	@echo "Running MCCL reduce module…"
	@echo ${MPIRUN} -np 8 $(MCCL_REDUCE_BIN)
	@$(MPIRUN) -np 8 $(MCCL_REDUCE_BIN)

.PHONY: mccl_reducescatter
mccl_reducescatter: $(MCCL_REDUCESCATTER_BIN)
	@echo "Running MCCL reducescatter module…"
	@echo ${MPIRUN} -np 8 $(MCCL_REDUCESCATTER_BIN)
	@$(MPIRUN) -np 8 $(MCCL_REDUCESCATTER_BIN)	

# ------------------------------------------------------------------------
# Compile and run MPI collective modules
# ------------------------------------------------------------------------
.PHONY: mpi_broadcast
mpi_broadcast: $(MPI_BROADCAST_BIN)
	@echo "Running MPI broadcast module with 8 processes..."
	@echo ${MPIRUN} -np 8 $(MPI_BROADCAST_BIN)
	@$(MPIRUN) -np 8 $(MPI_BROADCAST_BIN)

.PHONY: mpi_allgather
mpi_allgather: $(MPI_ALLGATHER_BIN)
	@echo "Running MPI allgather module with 8 processes..."
	@echo ${MPIRUN} -np 8 $(MPI_ALLGATHER_BIN)
	@$(MPIRUN) -np 8 $(MPI_ALLGATHER_BIN)

.PHONY: mpi_allreduce
mpi_allreduce: $(MPI_ALLREDUCE_BIN)
	@echo "Running MPI allreduce module with 8 processes..."
	@echo ${MPIRUN} -np 8 $(MPI_ALLREDUCE_BIN)
	@$(MPIRUN) -np 8 $(MPI_ALLREDUCE_BIN)

.PHONY: mpi_reduce
mpi_reduce: $(MPI_REDUCE_BIN)
	@echo "Running MPI reduce module with 8 processes..."
	@echo ${MPIRUN} -np 8 $(MPI_REDUCE_BIN)
	@$(MPIRUN) -np 8 $(MPI_REDUCE_BIN)

.PHONY: mpi_reducescatter
mpi_reducescatter: $(MPI_REDUCESCATTER_BIN)
	@echo "Running MPI reducescatter module with 8 processes..."
	@echo ${MPIRUN} -np 8 $(MPI_REDUCESCATTER_BIN)
	@$(MPIRUN) -np 8 $(MPI_REDUCESCATTER_BIN)	

# ------------------------------------------------------------------------
# Rules to compile kernels
# ------------------------------------------------------------------------
$(REDUCE_BIN): $(REDUCE_SRC) $(TIMER_SRC) $(DATA_SRC)
	@echo "Building reduction module…"
	@mkdir -p $(BUILD_DIR)/kernels
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(AXPY_BIN): $(AXPY_SRC) $(TIMER_SRC) $(DATA_SRC)
	@echo "Building axpy module…"
	@mkdir -p $(BUILD_DIR)/kernels
	$(NVCC) $(NVCCFLAGS) $(CUBLAS_LIBS) -o $@ $^

$(DOT_BIN): $(DOT_SRC) $(TIMER_SRC) $(DATA_SRC)
	@echo "Building dot product module…"
	@mkdir -p $(BUILD_DIR)/kernels
	$(NVCC) $(NVCCFLAGS) $(CUBLAS_LIBS) -o $@ $^

$(ADD_BIN): $(ADD_SRC) $(TIMER_SRC) $(DATA_SRC)
	@echo "Building add module…"
	@mkdir -p $(BUILD_DIR)/kernels
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(GEMM_BIN): $(GEMM_SRC) $(TIMER_SRC) $(DATA_SRC)
	@echo "Building GEMM module…"
	@mkdir -p $(BUILD_DIR)/kernels
	$(NVCC) $(NVCCFLAGS) $(CUBLAS_LIBS) -o $@ $^

# ------------------------------------------------------------------------
# Rules to compile MCCL collective modules
# ------------------------------------------------------------------------
$(MCCL_BROADCAST_BIN): $(MCCL_BROADCAST_SRC) $(MCCL_SRC) $(DATA_SRC)
	@echo "Building MCCL broadcast module…"
	@mkdir -p $(BUILD_DIR)/collectives/mccl
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -o $@ $^ 

$(MCCL_ALLGATHER_BIN): $(MCCL_ALLGATHER_SRC) $(MCCL_SRC) $(DATA_SRC)
	@echo "Building MCCL allgather module…"
	@mkdir -p $(BUILD_DIR)/collectives/mccl
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -o $@ $^ 

$(MCCL_ALLREDUCE_BIN): $(MCCL_ALLREDUCE_SRC) $(MCCL_SRC) $(DATA_SRC)
	@echo "Building MCCL allreduce module…"
	@mkdir -p $(BUILD_DIR)/collectives/mccl
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -o $@ $^ 

$(MCCL_REDUCE_BIN): $(MCCL_REDUCE_SRC) $(MCCL_SRC) $(DATA_SRC)
	@echo "Building MCCL reduction collective module…"
	@mkdir -p $(BUILD_DIR)/collectives/mccl
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -o $@ $^ 

$(MCCL_REDUCESCATTER_BIN): $(MCCL_REDUCESCATTER_SRC) $(MCCL_SRC) $(DATA_SRC)
	@echo "Building MCCL reducescatter module…"
	@mkdir -p $(BUILD_DIR)/collectives/mccl
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -o $@ $^

# ------------------------------------------------------------------------
# Rules to compile MPI collective modules
# ------------------------------------------------------------------------
$(MPI_BROADCAST_BIN): $(MPI_BROADCAST_SRC) $(DATA_SRC)
	@echo "Building MPI broadcast module…"
	@mkdir -p $(BUILD_DIR)/collectives/mpi
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -o $@ $^

$(MPI_ALLGATHER_BIN): $(MPI_ALLGATHER_SRC) $(DATA_SRC)
	@echo "Building MPI allgather module…"
	@mkdir -p $(BUILD_DIR)/collectives/mpi
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -o $@ $^

$(MPI_ALLREDUCE_BIN): $(MPI_ALLREDUCE_SRC) $(DATA_SRC)
	@echo "Building MPI allreduce module…"
	@mkdir -p $(BUILD_DIR)/collectives/mpi
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -o $@ $^

$(MPI_REDUCE_BIN): $(MPI_REDUCE_SRC) $(DATA_SRC)
	@echo "Building MPI reduce module…"
	@mkdir -p $(BUILD_DIR)/collectives/mpi
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -o $@ $^

$(MPI_REDUCESCATTER_BIN): $(MPI_REDUCESCATTER_SRC) $(DATA_SRC)
	@echo "Building MPI reducescatter module…"
	@mkdir -p $(BUILD_DIR)/collectives/mpi
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -o $@ $^

# ------------------------------------------------------------------------
# Cleaning up build files
# ------------------------------------------------------------------------
.PHONY: clean
clean:
	@echo "Cleaning up build files…"
	@echo rm -rf $(BUILD_DIR)
	@rm -rf $(BUILD_DIR)

# ------------------------------------------------------------------------
# End of Makefile
# ------------------------------------------------------------------------

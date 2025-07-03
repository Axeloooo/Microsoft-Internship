# =============================================================
#  Makefile – developer shortcuts for MIGC
# =============================================================

BUILD_DIR   ?= build
BUILD_TYPE  ?= Release
CUDA_ARCH   ?= 90
TARGET      ?= axpy

.PHONY: help build run test doc clean

# -------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  make build                - configure & compile (default Release)"
	@echo "  make run TARGET=<exe>     - run ./$(BUILD_DIR)/<exe>"
	@echo "  make test                 - run unit tests (ctest)"
	@echo "  make doc                  - build Doxygen docs"
	@echo "  make clean                - delete $(BUILD_DIR)/"

# -------------------------------------------------------------
build:
	@echo ">>> CMake configure + build"
	@cmake -S . -B $(BUILD_DIR) \
	    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
	@cmake --build $(BUILD_DIR) --parallel

# -------------------------------------------------------------
blas: build
	@if [ ! -f $(BUILD_DIR)/$(TARGET) ]; then \
	    echo "Binary '$(TARGET)' not found. Try another TARGET or run make build."; \
	    exit 1; \
	fi
	@echo ">>> Running blas ./$(BUILD_DIR)/$(TARGET)"
	@./$(BUILD_DIR)/$(TARGET)

# -------------------------------------------------------------
collective: build
	@if [ ! -f $(BUILD_DIR)/$(TARGET) ]; then \
	    echo "Binary '$(TARGET)' not found. Try another TARGET or run make build."; \
	    exit 1; \
	fi
	@echo ">>> Running collective ./$(BUILD_DIR)/$(TARGET)"
	@mpirun -np 8 ./$(BUILD_DIR)/$(TARGET) --elts 4194304 --dtype fp32 --iters 10

# -------------------------------------------------------------
test: build
	@echo ">>> Running unit tests"
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

# -------------------------------------------------------------
doc: build
	@echo ">>> Building docs"
	@cmake --build $(BUILD_DIR) --target doc_doxygen

# -------------------------------------------------------------
clean:
	@echo ">>> Cleaning up $(BUILD_DIR)/"
	@rm -rf $(BUILD_DIR)

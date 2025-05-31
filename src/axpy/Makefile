.PHONY: build
build:
	@echo "Building axpy module..."
	mkdir -p ../../build
	nvcc -I ../../include -arch=sm_90 -o ../../build/axpy main.cu -lcublas

.PHONY: run
run: build
	@echo "Running axpy module..."
	../../build/axpy

.PHONY: clean
clean:
	@echo "Cleaning up build files..."
	rm -rf ../../build/axpy
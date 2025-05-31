.PHONY: build
build:
	@echo "Building reduction module..."
	mkdir -p ../../build
	nvcc -I ../../include -arch=sm_90 -o ../../build/reduction main.cu

.PHONY: run
run: build
	@echo "Running reduction module..."
	../../build/reduction

.PHONY: clean
clean:
	@echo "Cleaning up build files..."
	rm -rf ../../build/reduction
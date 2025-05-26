.PHONY: build
build:
	@echo "Building vector_add module..."
	mkdir -p ../../build
	nvcc -I ../../include -o ../../build/vector_add main.cu

.PHONY: run
run: build
	@echo "Running vector_add module..."
	../../build/vector_add

.PHONY: clean
clean:
	@echo "Cleaning up build files..."
	rm -rf ../../build/vector_add
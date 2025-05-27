.PHONY: build
build:
	@echo "Building dot product module..."
	mkdir -p ../../build
	nvcc -I ../../include -o ../../build/dot_product main.cu -lcublas

.PHONY: run
run: build
	@echo "Running dot product module..."
	../../build/dot_product

.PHONY: clean
clean:
	@echo "Cleaning up build files..."
	rm -rf ../../build/dot_product
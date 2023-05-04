CC=gcc
CXX=g++
NVCC=nvcc

BLASPATH=/opt/asn/apps/blas_gcc610_avx
CUDAPATH=/opt/asn/apps/cuda_11.7.0

CCFLAGS=-std=c11
CXXFLAGS=-std=c++11 -O4
NVCCFLAGS=-std=c++11

NVCCARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70

TIMERINCPATH=-I$(CUDAPATH)/include -ITimer/include
INCPATH=-Imatrix_mul/include -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(GTESTPATH)/include
LIBPATH=-L$(CUDAPATH)/lib64 -L$(GTESTPATH)/lib64
RPATH=-Wl,-rpath=`pwd`/build/lib -Wl,-rpath=`pwd`/$(GTESTPATH)/lib64 -Wl,-rpath=`pwd`/$(CUDAPATH)/lib64
LIBS=-lcudart

.PHONY: clean modules
all: build/lib/libTimer.so build/lib/libmatrix_mul.so build/bin/matrix_mul_test build/bin/timer_test

build/lib/libTimer.so: modules Timer/src/Timer.cpp
	@mkdir -p build/.objects/Timer
	$(CXX) $(CXXFLAGS) -c -fPIC -ITimer/include \
		-I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc\
		-o build/.objects/Timer/Timer.os Timer/src/Timer.cpp
	@mkdir -p build/lib
	$(CXX) -shared -o build/lib/libTimer.so build/.objects/Timer/* \
		-L$(CUDAPATH)/lib64 -lcudart_static
	@mkdir -p build/include
	@ln -sf ../../Timer/include/Timer.hpp build/include/Timer.hpp

build/lib/libmatrix_mul.so: modules matrix_mul/src/matrix_mul.cu
	@mkdir -p build/.objects/matrix_mul
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-Imatrix_mul/include -I$(CUDAPATH)/samples/common/inc \
		-I $(CUDAPATH)/include \
		-dc -o build/.objects/matrix_mul/matrix_mul.o \
		matrix_mul/src/matrix_mul.cu
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-dlink -o build/.objects/matrix_mul/matrix_mul-dlink.o build/.objects/matrix_mul/matrix_mul.o	
	mkdir -p build/lib
	$(CC) -shared -o build/lib/libmatrix_mul.so build/.objects/matrix_mul/* \
		-Wl,-rpath=$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib64 -lcudart
	@mkdir -p build/include
	@ln -sf ../../matrix_mul/include/matrix_mul.h build/include/matrix_mul.h

build/bin/matrix_mul_test: build/lib/libTimer.so build/lib/libmatrix_mul.so \
	matrix_mul/test/src/test.cpp
	@mkdir -p build/bin
	$(NVCC) -Ibuild/include -I$(CUDAPATH)/samples/common/inc \
	-ITimer/include -Imatrix_mul/include -I $(CUDAPATH)/include \
	-o build/bin/matrix_mul_test matrix_mul/test/src/test.cpp \
	-Xlinker \
	-Lbuild/lib -L$(CUDAPATH)/lib64 \
	-lTimer -lmatrix_mul -lcudart -lblas -lgfortran -lcublas

build/bin/timer_test: build/lib/libTimer.so \
	Timer/test/src/test.cpp
	@mkdir -p build/bin
	$(CXX) -Ibuild/include -I$(CUDAPATH)/samples/common/inc \
		-ITimer/include \
		-o build/bin/timer_test Timer/test/src/test.cpp \
		-Wl,-rpath=$(PWD)/build/lib \
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lTimer -lcudart

run: build/bin/matrix_mul_test
	rm -f *.nsys-rep matrix_mul_test.i* matrix_mul_test.o* core.*; \
	echo -ne "gpu\n1\n\n20gb\n1\nampere\nmatrix_mul_test\n" | \
		run_gpu .runTests.sh > /dev/null; \
	sleep 10; \
	tail -f matrix_mul_test.o*; \

clean:
	rm -rf build
	rm -f *nsys-rep
	rm -f matrix_mul_test.*

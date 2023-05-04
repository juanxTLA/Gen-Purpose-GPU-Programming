CC=gcc
CXX=g++
NVCC=nvcc

CUDAPATH=/opt/asn/apps/cuda_11.7.0

CCFLAGS=-std=c11
CXXFLAGS=-std=c++11 -O4
NVCCFLAGS=-std=c++11

NVCCARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70

TIMERINCPATH=-I$(CUDAPATH)/include -ITimer/include
INCPATH=-Ilapack/include -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(GTESTPATH)/include
LIBPATH=-L$(CUDAPATH)/lib64 -L$(GTESTPATH)/lib64
RPATH=-Wl,-rpath=`pwd`/build/lib -Wl,-rpath=`pwd`/$(GTESTPATH)/lib64 -Wl,-rpath=`pwd`/$(CUDAPATH)/lib64
LIBS=-lcudart

.PHONY: clean modules
all: build/lib/libTimer.so build/lib/liblapack.so build/bin/lapack_test build/bin/timer_test

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

build/lib/liblapack.so: modules lapack/src/lapack.cu
	@mkdir -p build/.objects/lapack
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-Ilapack/include -I$(CUDAPATH)/samples/common/inc \
		-I $(CUDAPATH)/include \
		-dc -o build/.objects/lapack/lapack.o \
		lapack/src/lapack.cu
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-dlink -o build/.objects/lapack/lapack-dlink.o build/.objects/lapack/lapack.o	
	mkdir -p build/lib
	$(CC) -shared -o build/lib/liblapack.so build/.objects/lapack/* \
		-Wl,-rpath=$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib64 -lcudart
	@mkdir -p build/include
	@ln -sf ../../lapack/include/lapack.h build/include/lapack.h


build/bin/lapack_test: build/lib/libTimer.so build/lib/liblapack.so lapack/include/Matrix.hpp\
	lapack/test/src/test.cpp
	@mkdir -p build/bin
	$(NVCC) -Ibuild/include -I$(CUDAPATH)/samples/common/inc \
	-ITimer/include -Ilapack/include -I $(CUDAPATH)/include \
	-o build/bin/lapack_test lapack/test/src/test.cpp \
	-Xlinker \
	-Lbuild/lib -L$(CUDAPATH)/lib64 \
	-lTimer -llapacke -lcudart -lgfortran -lcublas -lcusolver

build/bin/timer_test: build/lib/libTimer.so \
	Timer/test/src/test.cpp
	@mkdir -p build/bin
	$(CXX) -Ibuild/include -I$(CUDAPATH)/samples/common/inc \
		-ITimer/include \
		-o build/bin/timer_test Timer/test/src/test.cpp \
		-Wl,-rpath=$(PWD)/build/lib \
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lTimer -lcudart

run: build/bin/lapack_test
	rm -f *.nsys-rep lapack_test.i* lapack_test.o* core.*; \
	echo -ne "gpu\n1\n\n20gb\n1\nampere\nlapack_test\n" | \
		run_gpu .runTests.sh > /dev/null; \
	sleep 5; \
	tail -f lapack_test.o*; \


run-valgrind: build/bin/lapack_test
	# call "module load valgrind" first
	valgrind --leak-check=full build/bin/lapack_test

clean:
	rm -rf build
	rm -f *nsys-rep
	rm -f lapack_test.*
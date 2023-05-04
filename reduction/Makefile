CC=gcc
CXX=g++
NVCC=nvcc

CUDAPATH=/opt/asn/apps/cuda_11.7.0

CCFLAGS=-std=c11
CXXFLAGS=-std=c++11 -O4
NVCCFLAGS=-std=c++11

NVCCARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70

TIMERINCPATH=-I$(CUDAPATH)/include -ITimer/include
INCPATH=-Ireduction/include -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(GTESTPATH)/include
LIBPATH=-L$(CUDAPATH)/lib64 -L$(GTESTPATH)/lib64
RPATH=-Wl,-rpath=`pwd`/build/lib -Wl,-rpath=`pwd`/$(GTESTPATH)/lib64 -Wl,-rpath=`pwd`/$(CUDAPATH)/lib64
LIBS=-lcudart

.PHONY: clean modules
all: build/lib/libTimer.so build/lib/libreduction.so build/bin/reduction_test build/bin/timer_test

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

build/lib/libreduction.so: modules reduction/src/reduction.cu
	@mkdir -p build/.objects/reduction
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-Ireduction/include -I$(CUDAPATH)/samples/common/inc \
		-I $(CUDAPATH)/include \
		-dc -o build/.objects/reduction/reduction.o \
		reduction/src/reduction.cu
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-dlink -o build/.objects/reduction/reduction-dlink.o build/.objects/reduction/reduction.o	
	mkdir -p build/lib
	$(CC) -shared -o build/lib/libreduction.so build/.objects/reduction/* \
		-Wl,-rpath=$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib64 -lcudart
	@mkdir -p build/include
	@ln -sf ../../reduction/include/reduction.h build/include/reduction.h

build/bin/reduction_test: build/lib/libTimer.so build/lib/libreduction.so \
	reduction/test/src/test.cpp
	@mkdir -p build/bin
	$(CXX) -Ibuild/include -I$(CUDAPATH)/samples/common/inc \
		-ITimer/include -Ireduction/include \
		-o build/bin/reduction_test reduction/test/src/test.cpp \
		-Wl,-rpath=$(PWD)/build/lib \
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lTimer -lreduction -lcudart

build/bin/timer_test: build/lib/libTimer.so \
	Timer/test/src/test.cpp
	@mkdir -p build/bin
	$(CXX) -Ibuild/include -I$(CUDAPATH)/samples/common/inc \
		-ITimer/include \
		-o build/bin/timer_test Timer/test/src/test.cpp \
		-Wl,-rpath=$(PWD)/build/lib \
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lTimer -lcudart

run: build/bin/reduction_test
	rm -f *.nsys-rep reduction_test.i* reduction_test.o* core.*; \
	echo -ne "class\n1\n\n1gb\n1\nampere\nreduction_test\n" | \
		run_gpu .runTests.sh > /dev/null; \
	sleep 5; \
	tail -f reduction_test.o*; \

clean:
	rm -rf build
	rm -f *nsys-rep
	rm -f reduction_test.*
	rm -f *.txt
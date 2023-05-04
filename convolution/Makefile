CC=gcc
CXX=g++
NVCC=nvcc

CUDAPATH=/opt/asn/apps/cuda_11.7.0

CCFLAGS=-std=c11
CXXFLAGS=-std=c++11 -O4
NVCCFLAGS=-std=c++11

NVCCARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70

TIMERINCPATH=-I$(CUDAPATH)/include -ITimer/include
INCPATH=-Iconvolution/include -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(GTESTPATH)/include
LIBPATH=-L$(CUDAPATH)/lib64 -L$(GTESTPATH)/lib64
RPATH=-Wl,-rpath=`pwd`/build/lib -Wl,-rpath=`pwd`/$(GTESTPATH)/lib64 -Wl,-rpath=`pwd`/$(CUDAPATH)/lib64
LIBS=-lcudart

.PHONY: clean modules
all: build/lib/libTimer.so build/lib/libconvolution.so build/bin/convolution_test build/bin/timer_test

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

build/lib/libconvolution.so: modules convolution/src/convolution.cu
	@mkdir -p build/.objects/convolution
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-Iconvolution/include -I$(CUDAPATH)/samples/common/inc \
		-I $(CUDAPATH)/include \
		-dc -o build/.objects/convolution/convolution.o \
		convolution/src/convolution.cu
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-dlink -o build/.objects/convolution/convolution-dlink.o build/.objects/convolution/convolution.o	
	mkdir -p build/lib
	$(CC) -shared -o build/lib/libconvolution.so build/.objects/convolution/* \
		-Wl,-rpath=$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib64 -lcudart
	@mkdir -p build/include
	@ln -sf ../../convolution/include/convolution.h build/include/convolution.h

build/bin/convolution_test: build/lib/libTimer.so build/lib/libconvolution.so \
	convolution/test/src/test.cpp
	@mkdir -p build/bin
	$(CXX) -Ibuild/include -I$(CUDAPATH)/samples/common/inc \
		-ITimer/include -Iconvolution/include \
		-o build/bin/convolution_test convolution/test/src/test.cpp \
		-Wl,-rpath=$(PWD)/build/lib \
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lTimer -lconvolution -lcudart -lgfortran

build/bin/timer_test: build/lib/libTimer.so \
	Timer/test/src/test.cpp
	@mkdir -p build/bin
	$(CXX) -Ibuild/include -I$(CUDAPATH)/samples/common/inc \
		-ITimer/include \
		-o build/bin/timer_test Timer/test/src/test.cpp \
		-Wl,-rpath=$(PWD)/build/lib \
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lTimer -lcudart

run: build/bin/convolution_test
	rm -f *.nsys-rep convolution_test.i* convolution_test.o* core.*; \
	echo -ne "gpu\n1\n\n20gb\n1\nampere\nconvolution_test\n" | \
		run_gpu .runTests.sh > /dev/null; \
	sleep 5; \
	tail -f convolution_test.o*; \

clean:
	rm -rf build
	rm -f *nsys-rep
	rm -f convolution_test.*

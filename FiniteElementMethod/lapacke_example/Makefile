.PHONY: clean

test: test.o vector.o
	# call "module load lapack" first
	g++ -std=c++11 -o test test.o vector.o -llapacke

vector.o: vector.hpp vector.cpp
	g++ -c -std=c++11 vector.cpp

test.o: vector.hpp test.cpp
	g++ -c -std=c++11 test.cpp

run: test
	./test

run-valgrind: test
	# call "module load valgrind" first
	valgrind --leak-check=full ./test

clean:
	rm -rf test

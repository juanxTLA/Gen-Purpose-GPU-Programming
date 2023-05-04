#ifndef __vector_hpp
#define __vector_hpp

#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

using std::endl;
using std::cout;
using std::vector;

class Vector{

    public:
        vector<float> vect;
        int size;

        Vector();
        Vector(vector<float> a, int n);
        Vector(int n);

        float*  getVector();
        float   getMod();

        void normalize();
        void push(float a);
        void insert(int i, float v);
        void setVector(vector<float> v, int n);
        void print();

        ~Vector();

        float operator[](int i);
        float operator*(Vector a);
        

        Vector operator-(Vector a);
        Vector operator+(Vector a);
        Vector operator*(float a);
        Vector operator/(float a);
};

#endif
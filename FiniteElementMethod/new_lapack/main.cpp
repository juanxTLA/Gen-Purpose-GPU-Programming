#include <lapacke.h>
#include "Timer.hpp"
#include <cstdio>
#include <vector>
#include <iostream>

#include <Matrix.hpp>
#include <memory>

#define PI 3.14159

namespace  Quadrature {
  static const std::vector<double> weights {
    2.089795918367347e-01,
    1.909150252525595e-01,
    1.909150252525595e-01,
    1.398526957446383e-01,
    1.398526957446383e-01,
    6.474248308443485e-02,
    6.474248308443485e-02,
  };
  static const std::vector<double> points {
    5.000000000000000e-01,
    7.029225756886985e-01,
    2.970774243113014e-01,
    1.292344072003028e-01,
    8.707655927996972e-01,
    2.544604382862076e-02,
    9.745539561713792e-01,
  };
  template <typename Function>
  static double integrate(Function const & aFunction) {
    double integral = 0.0;
    for (int idx = 0; idx < points.size(); ++idx) {
      integral += weights[idx] * aFunction(points[idx]);
    }
    return integral;
  }
};

void fem1d(Matrix<float> &x);

float intHat1(float x1, float x2);
float intHat2(float floatx1, float x2);

float hat1(float x, float x1, float x2);
float hat2(float x, float x1, float x2);

float f(float x);

using namespace std;

int main(int argc, char* argv[]) {

  auto f = [] (double x) {
    return 8;
  };

  int sizeOfMatrix = 7;
  int numberOfRightHandSides = 1;
  int leadingDimensionOfA = sizeOfMatrix;
  int leadingDimensionOfB = numberOfRightHandSides;

  int n = atoi(argv[1]);
  float step = 1.0f/(n-1);
  // Compute values
  Matrix<float> x(n, 1);

  for(int i = 0; i < n; ++i){
    x(i, 0) = i * step;
  }

  // Timer t1;

  // t1.start();
  fem1d(x);
  // t1.stop();

  // cout << n << " " << t1.elapsedTime_ms() << endl;

  auto integral = Quadrature::integrate([] (double x) {
    return 8 * (1 - x); 
  });

  //printf("integral value is %20.16f\n", integral);

  return 0;
}

void fem1d(Matrix<float> &x){
  int m = x.numberOfRows();

  Matrix<float> h(m-1, 1);

  for(int i = 0; i < m-1; ++i){
    h(i, 0) = x(i+1, 0) - x(i, 0);
  }

  Matrix<float> a(m, m);
  Matrix<float> f(m, 1);
  Matrix<int> ipiv(m, 1);

  a(0, 0) = 1.0f;
  a(1, 1) = 1/h(0,0);
  a(m-1, m-1) = 1.0f;

  f(m-1, 0) = 0.0f;
  f(1, 0) = intHat1(x(0,0), x(1,0));

  for(int i = 1; i < m - 2; ++i){
    a(i,i)      = a(i,i) + 1/h(i,0);
    a(i,i+1)    = a(i,i+1) - 1/h(i,0);
    a(i+1,i)    = a(i+1,i) - 1/h(i,0);
    a(i+1,i+1)  = a(i+1,i+1) + 1/h(i,0);

    f(i, 0) = f(i,0) + intHat2(x(i,0), x(i+1,0));
    f(i+1, 0) = intHat1(x(i,0), x(i+1,0));
  }

  a(m-2,m-2) = a(m-2,m-2) + 1/h(m-2,0);
  f(m-2, 0) = f(m-2,0) + intHat2(x(m-2,0), x(m-1,0));

  //we need to solve for A \ F

  LAPACKE_sgesv(
    LAPACK_ROW_MAJOR,
    a.numberOfRows(),
    f.leadingDimension(),
    a.data(),
    a.leadingDimension(),
    ipiv.data(),
    f.data(),
    f.leadingDimension()
  );

  f.print();

}


float intHat1(float x1, float x2){
    float xm = (x1 + x2)*0.5;
    float y = (x2 - x1) * (f(x1) * hat1(x1, x1, x2) + 4*f(xm) * hat1(xm, x1, x2) +
                    f(x2) * hat1(x2,x1,x2)) / 6;

    return y;
}

float intHat2(float x1, float x2){
    float xm = (x1 + x2)*0.5;
    float y = (x2 - x1) * (f(x1) * hat2(x1, x1, x2) + 4*f(xm) * hat2(xm, x1, x2) +
                    f(x2) * hat2(x2,x1,x2)) / 6;

    return y;
}

float hat1(float x, float x1, float x2){
    return (x-x1)/(x2-x1);
}

float hat2(float x, float x1, float x2){
    return (x2-x)/(x2-x1);
}

float f(float x){
    return PI*PI*sin(PI*x);
}                                                                                           
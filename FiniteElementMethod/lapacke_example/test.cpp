#include <cstdio>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <lapacke.h>

#include "vector.hpp"

#define PI 3.14159
#define DEBUG

using std::cout;
using std::endl;
using std::ifstream;

float intHat1(float x1, float x2);
float intHat2(float floatx1, float x2);

float hat1(float x, float x1, float x2);
float hat2(float x, float x1, float x2);
float f(float x);

ma
Vector gmres(Vector a[], Vector b, Vector x0, int k, float tol);

Vector getCol(int j, Vector x[], int rows);
void setCol(int j, Vector x[], int rows, Vector val);

void printMatrix(Vector a[], int row);

Vector operator*(Vector a[], Vector b);

Vector lapackCall(Vector a[], Vector &b, int aRows, int aCols);

int main(int argc, char* argv[]) {

  int nodalPoints = atoi(argv[0]);

  vector<float> _x;
  Vector x = Vector(_x, 7);
  x.push(0.0f);
  x.push(1/7);
  x.push(0.3f);
  x.push(0.333f);
  x.push(0.5f);
  x.push(0.75f);
  x.push(1.0f);

  Vector sol = fem1d(x);

  sol.print();

  return 0;
}

Vector fem1d(Vector &x){
  int m = x.size;

  vector<float> _h;
  Vector h = Vector(_h, m-1);

  for(int i = 0; i < h.size; i++){
    h.push(x.vect[i+1] - x.vect[i]);
  }

  Vector* a = new Vector[m];

  //initialize the array with 0s
  for(int i = 0; i < m; i++){
    vector<float>_ai(m, 0.0f);
    a[i].setVector(_ai, m);
  }
  
  vector<float> _f(m, 0.0f);
  Vector f = Vector(_f, m);

  a[0].vect[0] = 1.0f;
  a[m-1].vect[m-1] = 1.0f;
  a[1].vect[1] = 1/h[0];

  f.vect[m-1] = 0.0f;
  f.vect[1] = intHat1(x[0], x[1]);

  for(int i = 1; i < m - 2; i++){
      a[i].vect[i] = a[i][i] + 1/h[i];
      a[i].vect[i+1] = a[i][i+1] - 1/h[i];
      a[i+1].vect[i] = a[i+1][i] - 1/h[i];
      a[i+1].vect[i+1] = a[i+1][i+1] + 1/h[i];

      f.vect[i] = f[i] + intHat2(x[i], x[i+1]);
      f.vect[i+1] = f[i+1] + intHat2(x[i], x[i+1]);
  }

  a[m-2].vect[m-2] = a[m-2][m-2] + 1/h[m-2];
  f.vect[m-2] = f[m-2] + intHat2(x[m-2], x[m-1]);

  #ifdef DEBUG
    for(int r = 0; r < m; r++){
      for(int c = 0; c < a[0].size; c++){
        if(a[r][c] != 0.0f){
            cout << "(" << r+1 << "," << c+1 << ") = " << a[r][c] << endl;
        } 
      }
    }

    cout << endl;

    for(int i = 0; i < h.size; i++){

      cout << h[i] << endl;
    }

    cout << endl << endl;
  #endif

  vector<float> _x0(m, 1.0f);
  Vector x0 = Vector(_x0, m);

  int sizeOfMatrix = m;
  int numberOfRightHandSides = 1;
  int leadingDimensionOfA = sizeOfMatrix;
  int leadingDimensionOfB = numberOfRightHandSides;

  //CALL LINEAR SOLVER (A \ F in MATLAB code)

  float *A = (float*)malloc(m*m*sizeof(float));
  int n = 0;
  for(int i = 0; i < m; ++i){
    memcpy(&A[n], a[i].getVector(), m*sizeof(float));
    n += m;
  }

  for(int i = 0; i < m*m; ++i){
    cout << A[i] << endl;
  }

  std::vector<float> _b(sizeOfMatrix * numberOfRightHandSides, 0);
  Vector b(_b, sizeOfMatrix);

  std::vector<int> ipiv(sizeOfMatrix);
  
  LAPACKE_sgesv(
    LAPACK_ROW_MAJOR,
    sizeOfMatrix,
    numberOfRightHandSides,
    A,
    leadingDimensionOfA,
    ipiv.data(),
    b.getVector(),
    leadingDimensionOfB
  );



  // Vector sol = gmres(a, f, x0, 3, 0.1);
  
  return b;
}

Vector gmres(Vector a[], Vector b, Vector x0, int k, float tol){
	
  Vector y;

	int qRows = b.size;
	int hRows = k+1;
  
	Vector* q = new Vector[qRows];
	for(int i = 0; i < b.size; i++){
    vector<float> _qi(k+1, 0.0f);
		q[i].setVector(_qi, k+1);
	}

	Vector* h = new Vector[hRows];
	for(int i = 0; i < k+1; i++){
    vector<float>_hi(k, 0.0f);
		h[i].setVector(_hi, k);
	}

  #ifdef DEBUG
      printMatrix(q, qRows);
  #endif

  Vector r0 = b - (a*x0);

  float beta = r0.getMod();
  r0.normalize();

  setCol(0, q, qRows, r0);

  vector<float> _e1(k+1, 0.0f);
  Vector e1 = Vector(_e1, k+1);
  e1.vect[0] =  1.0f;
	
  for(int j = 0; j < k; j++){
    cout << "j = " << j << endl;
    setCol(j+1, q, qRows, a*getCol(j,q,qRows));

    for(int i = 0; i <= j; i++){
      Vector c1 = getCol(i, q, qRows);
      Vector c2 = getCol(j+1, q, qRows);
      float t = c1 * c2;

      h[i].vect[j] = t;
      Vector temp = getCol(j+1, q, qRows) - (getCol(i, q, qRows)*h[i][j]);
      setCol(j+1, q, qRows, temp);
    }
    

    h[j+1].vect[j] = getCol(j+1, q, qRows).getMod();
    
    if(abs(h[j+1][j]) > tol){
      setCol(j+1, q, qRows, getCol(j+1, q, qRows)/h[j+1][j]);
    }
    
    Vector e1Beta = e1 * beta;

    #ifdef DEBUG
      printMatrix(q, qRows);
    #endif

    y = lapackCall(h, e1Beta, j+2, j+1);
    
    float res = y.getMod();

    // if(res < tol){
    //   return (q*y) + x0;
    // }
  }

  return (q*y) + x0;
  //return e1;	
}

Vector getCol(int j, Vector x[], int rows){
	vector<float> _col(rows, 0.0f);
  Vector col = Vector(_col, rows);

	for(int i = 0; i < rows; i++){
    col.vect[i] = x[i][j];
	}

	return col;
}

void setCol(int j, Vector x[], int rows, Vector val){
  for(int i = 0; i < rows; i++){
		x[i].vect[j] = val[i];
	}
}

Vector lapackCall(Vector a[], Vector &b, int aRows, int aCols){
  //prepare input to lapack function
  char trans = 'N';
  int matrix_layout = LAPACK_ROW_MAJOR;
  int bSize = aRows;
  float * bArr = b.getVector();

  float * arr = (float*)malloc(aRows * aCols * sizeof(float));
  int i = 0;
  for(int r = 0; r <= aRows; r++){
    for(int c = 0; c <= aCols; c++){
      arr[i] = a[r][c];
    }
  }

  #ifdef DEBUG
    cout << "aMAtrix: " << endl;
    printMatrix(a, aRows);
    cout << endl;
  #endif

  LAPACKE_sgels(
    matrix_layout,
    trans,
    aRows,
    aCols,
    1,
    arr,
    aCols,
    bArr,
    1
  );

  vector<float> _y(bArr, bArr + b.size);
  Vector y = Vector(_y, bSize);
  
  #ifdef DEBUG
    cout << "y: " << endl;
    y.print();
    cout << endl;
  #endif

  return y;
}

Vector operator*(Vector a[], Vector b){
  vector<float> _c;
	Vector c = Vector(_c, b.size);
	for(int i = 0; i < b.size; i++){
    c.push(a[i] * b);
	}

	return c;
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
}

void printMatrix(Vector a[], int row){
  for(int i = 0; i < row; i++){
    a[i].print();
  }
}
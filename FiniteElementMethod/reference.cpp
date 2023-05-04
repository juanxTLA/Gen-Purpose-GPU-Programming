#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <cstdlib>
#include <fstream>
#include <lapacke.h>

using std::vector;
using std::ifstream;
using std::cout;
using std::endl;

#define PI 3.14159

//LAPACKE_sgels for C version with no addresses

//#define DEBUG


struct Vector{
    vector<float> vect;
    
    int getSize(){
        return vect.size();
    }

    void normalize(){
        //find mod of b
        float sum = 0;
        for(int i = 0; i < vect.size(); i++){
            sum += vect[i] * vect[i];
        }

        sum = sqrt(sum);

        for(int i = 0; i < vect.size(); i++){
            vect[i] /= sum;
        }
        
    }

    float getMod(){
        float mod;

        for(int i = 0; i < vect.size(); i++){
            mod += vect[i] * vect[i];
        }

        return mod;
    }

    void push(float a){
        vect.push_back(a);
    }

    Vector operator-(Vector a){

        Vector c;

        if(a.getSize() != getSize()) std::cout << "ERROR: sizes must match" << std::endl;

        else{
            for(int i = 0; i < vect.size(); i++){
                c.push(vect[i] - a.vect[i]);
            }
        }

        return c;
    }

    Vector operator+(Vector a){

        Vector c;
        if(a.getSize() != getSize()) std::cout << "ERROR: sizes must match" << std::endl;

        else{
            for(int i = 0; i < vect.size(); i++){
                c.push(vect[i] + a.vect[i]);
            }
        }

        return c;
    }

    Vector operator*(float a){

        Vector c;

        for(int i = 0; i < vect.size(); i++){
            c.push(vect[i] * a);
        }
        

        return c;
    }    
    
    Vector operator/(float a){

        Vector c;

        for(int i = 0; i < vect.size(); i++){
            c.push(vect[i] / a);
        }
        

        return c;
    }

    float operator*(Vector a){

        float sum = 0.0f;
        if(a.getSize() != getSize()) std::cout << "ERROR: sizes must match" << std::endl;

        else{
            for(int i = 0; i < vect.size(); i++){
                sum += vect[i] * a.vect[i];
            }
        }

        return sum;
    }

};

Vector constructVector(vector<float> a);

struct Array{
    vector<vector<float> > array;

    void insertColumn(int j, Vector v){

        for(int r = 0; r < array[j].size(); r++){
            array[r][j] = v.vect[j];
        }
    }

    void transpose(){
        for(int r = 0; r < array.size(); r++){
            for(int c = r; c < array[0].size(); c++){
                float temp = array[r][c];
                array[r][c] = array[c][r];
                array[c][r] = temp;
            }
        }
    }

    Vector getCol(int j){
        vector<float> temp;

        for(int i = 0; i < array[0].size(); i++){
            temp.push_back(array[i][j]);
        }

        return constructVector(temp);
    }

};

inline Vector operator*(Array a, Vector x){
   
    Vector sol;
    
    for(int r = 0; r < a.array.size(); r++){
        float temp = 0.0f;
        for(int c = 0; c < a.array[0].size(); c++){
            temp += a.array[r][c] * x.vect[c];
        }

        sol.vect.push_back(temp);
    }

    return sol;
}

float intHat1(float x1, float x2);
float intHat2(float x1, float x2);

float hat1(float x, float x1, float x2);
float hat2(float x, float x1, float x2);
float f(float x);

vector <float> fem1d(vector<float> x);

Array constructArray(vector<vector<float> > a);

Vector GMRES(Array a, Vector b, Vector x0, int k, float tol);

int main(int argc, char* argv[]){
    //adapted from MATLAB code
    int nodalPoints = atoi(argv[0]);

    //for more flexibility
    ifstream inFile;
    inFile.open("initial_values.txt");

    //MATLAB hard-coded values
    float help[] = {0.0f, 0.1f, 0.3f, 0.333f, 0.5f, 0.75f, 1.0f};
    vector<float> x(help, help + sizeof(help) / sizeof(float));

    Vector a = constructVector(x);

    vector<float> s = fem1d(x);
    
}

vector <float> fem1d(vector<float>x){
    int m = x.size(); //get length of x
    
    vector <float> h;
    for(int i = 0; i < m - 1; i++){
        h.push_back(x[i+1] - x[i]);       
    }

    vector<vector<float> > A(m, vector<float>(m, 0)); //mimic sparse function in MATLAB
    vector<float> F(m, 0); //mimic zeros function in MATLAB

    A[0][0] = 1;
    A[m-1][m-1] = 1;
    A[1][1] = 1/h[0];

    F[0] = 0;
    F[m-1] = 0;
    F[1] = intHat1(x[0], x[1]);

    for(int i = 1; i < m - 2; i++){
        A[i][i]     = A[i][i] + 1/h[i];
        A[i][i+1]   = A[i][i+1] - 1/h[i];
        A[i+1][i]   = A[i+1][i] - 1/h[i];
        A[i+1][i+1] = A[i+1][i+1] + 1/h[i];

        F[i]        = F[i] + intHat2(x[i], x[i+1]);
        F[i+1]      = F[i+1] + intHat2(x[i], x[i+1]);
    }


    A[m-2][m-2] = A[m-2][m-2] + 1/h[m-2];
    F[m-2]      = F[m-2] + intHat2(x[m-2], x[x[m-1]]);

    #ifdef DEBUG
        for(int r = 0; r < A.size(); r++){
            for(int c = 0; c < A[0].size(); c++){
                if(A[c][r] != 0.0f){
                    cout << "(" << c + 1 << "," << r + 1 << ") = " << A[c][r] << endl;
                } 
            }
        }

        cout << endl;

        for(int i = 0; i < h.size(); i++){
            cout << h[i] << endl;
        }
    #endif
    //solve system of equations GMRES
    //normalize F
    vector<float> x0Temp(m, 1.0f); //initial guess
    Vector x0 = constructVector(x0Temp);
    Array a = constructArray(A);

    Vector b = constructVector(F);
    b.normalize();

    int k = 1000;
    float tol = 0.1;
    
    //TODO: CALL GMRES FUNCTION   
    return h;
}

Vector GMRES(Array a, Vector b, Vector x0, int k, float tol){
    
    vector<vector<float> > Q;
    Array q = constructArray(Q);

    vector<vector<float> > H(k+1, vector<float>(k, 0));
    Array h = constructArray(H);

    Vector r0 = b - a*x0;
    r0.normalize();

    q.insertColumn(0, r0);

    float beta = r0.getMod();
    
    vector<float> E1(k + 1, 0.0f);
    E1[0] = 1.0f;
    Vector e1 = constructVector(E1);

    for(int j = 0; j < k; j++){
        Vector temp = q.getCol(0);
        q.insertColumn(j+1, a * temp);

        for(int i = 0; i <= j; i++){
            h.array[i][j] = q.getCol(i) * q.getCol(j+1);
            q.insertColumn(j+1, q.getCol(j+1) - (q.getCol(i) * h.array[i][j]));
        }

        h.array[j+1][j] = q.getCol(j+1).getMod();

        if(abs(h.array[j+1][j]) > tol){
            q.insertColumn(j+1, q.getCol(j+1)/h.array[j+1][j]);
        }

        

    }
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

Vector constructVector(vector<float> a){
    Vector x;
    x.vect = a;

    return x;
}

Array constructArray(vector<vector<float> > a){
    Array x;
    x.array = a;

    return x;
}

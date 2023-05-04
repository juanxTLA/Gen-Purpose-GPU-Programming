#include "vector.hpp"

Vector::Vector(){}

Vector::Vector(vector<float> a, int n){
    size = n;
    vect = a;
}

float* Vector::getVector(){
    return vect.data();
}

void Vector::normalize(){
    float sum = 0;

    for(int i = 0; i < size; i++){
        sum += vect[i] * vect[i];
    }

    sum = sqrt(sum);

    for(int i = 0; i < size; i++){
        vect[i] /= sum;
    }
}

float Vector::getMod(){
    float mod = 0.0f;

    for(int i = 0; i < size; i++){
        mod += vect[i] * vect[i];
    }

    return sqrt(mod);
}

void Vector::push(float a){
    vect.push_back(a);
}

void Vector::insert(int i, float v){
    vect[i] = v;
}

void Vector::setVector(vector<float> v, int n){
    vect = v;
    size = n;
}

void Vector::print(){
    for(int i = 0; i < size; i++){
        cout << vect[i] << " ";
    }

    cout << endl;
}

Vector::~Vector(){
    
}

float Vector::operator[](int i){
    return vect[i];
}

Vector Vector::operator+(Vector a){
    vector<float> _c(size, 0.0f);
    Vector c = Vector(_c, size);
    
    for(int i = 0; i < size; i++){
        c.push(vect[i] + a[i]);
    }

    return c;
}

Vector Vector::operator-(Vector a){
    vector<float> _c;
    Vector c = Vector(_c, size);
    
    for(int i = 0; i < size; i++){
        c.push(vect[i] - a[i]);
    }

    return c;
}

Vector Vector::operator*(float a){
    vector<float> _c;
    Vector c = Vector(_c, size);
    
    for(int i = 0; i < size; i++){
        c.push(vect[i] * a);
    }

    return c;
}

Vector Vector::operator/(float a){
    vector<float> _c;
    Vector c = Vector(_c, size);
    
    for(int i = 0; i < size; i++){
        c.push(vect[i] / a);
    }

    return c;
}

float Vector::operator*(Vector a){
    float res = 0.0f;
    for(int i = 0; i < vect.size(); i++){
        res += vect[i] * a[i];
    }
    return res;
}
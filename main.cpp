#include <stdio.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <string>
#include <time.h>
#include <random>
#include <vector>

using namespace std;

void randdouble(double min, double max, double *arr, int n){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(min, max);
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        arr[i] = dis(gen);
    }
}

double g(double x, double y){
    return pow(x+y,2)/x;
}

double f(double x){
  return (pow (x, 2) + sin (0.48 * (x + 2))) / (exp (pow (x, 2)) + 0.38);
}

double mc_method(double a, double b, int n, int k){
    double *x = new double[n], *y = new double[n];
    randdouble(0.4, 1.0, x, n);
    randdouble(0.0, 1.0, y, n);
    int in = 0;
    #pragma omp parallel for reduction(+:in)
    for (int i = 0; i < n; i++){
        if (y[i] < f(x[i])) in++;
    }
    return 0.6 * in/n;
}

double nc_method(double a, double b, int n, int k){
    double S1 = 0.0, S2 = 0.0, step = (b-a)/n;
    #pragma omp parallel for reduction(+:S1)
    for (int i = 1; i < n; i++){ if (i % 3 == 0) S1 += f(a + i*step); }
    #pragma omp parallel for reduction(+:S2)
    for (int i = 1; i < n; i++){ if (i % 3 != 0) S2 += f(a + i*step); }
    return 3./8. * (f(a) + f(b) + 2*S1 + 3*S2) * step;
}

double simpson_method(double a, double b, int n, int k){
    double S1 = 0.0, S2 = 0.0, step = (b-a)/n;
    #pragma omp parallel for reduction(+:S1)
    for (int i = 2; i < n; i+=2) S1 += f(a + i*step);
    #pragma omp parallel for reduction(+:S2)
    for (int i = 1; i < n - 1; i+=2) S2 += f(a + i*step);
    return (f(a) + f(b) + 2*S1 + 4*S2) * step/3.;
}

double trap_method(double a, double b, int n, int k){
    double S = 0.0, step = (b-a)/n;
    double diff = (f(a) + f(b)) / 2;
    #pragma omp parallel for reduction(+:S)
    for (int i = 1; i < n - 1; i++) S += f(a + i*step);
    return (diff + S) * step;
}

double rect_method(double a, double b, int n, int k){
    double S = 0.0, h = (b-a)/n;
    #pragma omp parallel for reduction(+:S)
    for (int i = k%2; i < n - (k%3); i++) S += f(a + i*h + (k%5)*h/2);
    return S * h;
}

void calc(double (*ptr)(double, double, int, int), double a, double b, double eps, int k){
  clock_t start = clock();
  double res, delta = 1.0;
  int n = 100000, coef = (k == 0) ? 15. : 3.;
  while (eps < delta){
    res = (*ptr)(a,b,n,k);
    delta = (*ptr)(a,b,2*n,k) - res;
    n *= 2;
  }
  cout << res << " " << fabs(delta/coef) << " " <<(double)(clock() - start) << " ms" << endl;
}

int main(int argc, char* argv[]){
  omp_set_num_threads(4);
  double eps = 0.0001;
  double a = 0.4, b = 1.0;
  cout << "Result    " << "Delta       "  << "Time" << endl;
  calc(&rect_method, a, b, eps, 10);
  calc(&rect_method, a, b, eps, 15);
  calc(&rect_method, a, b, eps, 16);
  calc(&trap_method, a, b, eps, -1);
  calc(&simpson_method, a, b, eps, 0);
  calc(&nc_method, a, b, eps, -2);
  calc(&mc_method, a, b, eps, -2);
  return 0;
}

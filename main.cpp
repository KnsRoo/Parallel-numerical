#include <stdio.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <string>
#include <time.h>

using namespace std;

double
f (double x){
  return (pow (x, 2) + sin (0.48 * (x + 2))) / (exp (pow (x, 2)) + 0.38);
}

double nc_method(double a, double b, int n, int k){
    double S = 0.0, step = (b-a)/n, m = f(a) + f(b);
    #pragma omp parallel for reduction(+:S)
    for (int i = 1; i < n - 1; i++){
            S += f(a + i*step + 1./3.*step) + f(a + i*step + 2./3.*step);
    }
    return 3./8. * (b-a) * (m/3. + S);
}

double simpson_method(double a, double b, int n, int k){
    double S1 = 0.0, S2 = 0.0, step = (b-a)/n, m = f(a) + f(b);
    #pragma omp parallel for
    for (int i = 1; i < n - 1; i++){
        if (i % 2 == 0) S1 += f(a + i*step);
        else S2 += f(a + i*step);
    }
    return (m + 2*S1 + 4*S2) * step/3.;
}

double trap_method(double a, double b, int n, int k){
    double S = 0.0, step = (b-a)/n;
    double diff = (f(a) + f(b)) / 2;
    #pragma omp parallel for reduction(+:S)
    for (int i = 1; i < n - 1; i++){
        S += f(a + i*step);
    }
    return (diff + S) * step;
}

double rect_method(double a, double b, int n, int k){
    double S = 0.0, h = (b-a)/n;
    #pragma omp parallel for reduction(+:S)
    for (int i = k%2; i < n - (k%3); i++){
        S += f(a + i*h + (k%5)*h/2);
    }
    return S * h;
}

void calc(double (*ptr)(double, double, int, int), double a, double b, int n, int k){
  clock_t start = clock();
  double res = (*ptr)(a,b,n,k);
  double res1 = (*ptr)(a,b,2*n,k);
  int coef = (k == 0) ? 15 : 3;
  double delta = fabs(res1-res)/coef;
  cout << res << " " << delta << " " <<(double)(clock() - start) << " ms" << endl;
}


int main(int argc, char* argv[]){
  omp_set_num_threads(4);
  int n = 100000;
  double a = 0.4, b = 1.0;
  cout << "Result    " << "Delta       "  << "Time" << endl;
  calc(&rect_method, a, b, n, 10);
  calc(&rect_method, a, b, n, 15);
  calc(&rect_method, a, b, n, 16);
  calc(&trap_method, a, b, n, -1);
  calc(&simpson_method, a, b, n, 0);
  calc(&nc_method, a, b, 3, -1);
  return 0;
}

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <string>
#include <time.h>

using namespace std;

double f(double x){
  return (pow (x, 2) + sin (0.48 * (x + 2))) / (exp (pow (x, 2)) + 0.38);
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
    for (int i = k%2; i < n - (k%3); i++){
        S += f(a + i*h + (k%5)*h/2);
    }
    return S * h;
}

void calc(double (*ptr)(double, double, int, int), double a, double b, double eps, int k){
  clock_t start = clock();
  double res, res1, delta = 1.0;
  int n = 1000, coef = (k == 0) ? 15. : 3.;
  while (eps < delta){
    res = (*ptr)(a,b,n,k);
    delta = (*ptr)(a,b,2*n,k) - res;
    n *= 2;
  }
  cout << res << " " << fabs(delta/coef) << " " <<(double)(clock() - start) << " ms" << endl;
}

int main(int argc, char* argv[]){
  omp_set_num_threads(4);
  double eps = 0.00001;
  double a = 0.4, b = 1.0;
  cout << "Result    " << "Delta       "  << "Time" << endl;
  calc(&rect_method, a, b, eps, 10);
  calc(&rect_method, a, b, eps, 15);
  calc(&rect_method, a, b, eps, 16);
  calc(&trap_method, a, b, eps, -1);
  calc(&simpson_method, a, b, eps, 0);
  calc(&nc_method, a, b, eps, -2);
  return 0;
}

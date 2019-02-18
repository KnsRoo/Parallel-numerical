#include <stdio.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <string>
#include <time.h>
#include <random>

typedef double(*function)(double, double);

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
    return  ((y < x) and (y > x/2)) ? pow(x+y,2)/x : 0;
}

double f(double arg, double x){
  return (pow (x, 2) + sin (0.48 * (x + 2))) / (exp (pow (x, 2)) + 0.38);
}

double mc_method_x(function f, double arg, double a, double b, double c, double d, int n){
    double *x = new double[n], *y = new double[n]; int in = 0;
    randdouble(a, b, x, n);
    randdouble(c, d, y, n);
    #pragma omp parallel for reduction(+:in)
    for (int i = 0; i < n; i++){ if (y[i] < f(arg, x[i])) in++; }
    return (b-a)*(d-c) * in/n;
}

double mc_method_xy(function f, double a, double b, double c, double d, int n){
    double S = 0.0;
    double *x = new double[n], *y = new double[n];
    randdouble(a, b, x, n);
    randdouble(c, d, y, n);
    #pragma omp parallel for reduction(+:S)
    for (int i = 0; i < n; i++) { S+=g(x[i],y[i]);}
    return ((b-a) * (d-c))/n * S;
}

double nc_method_x(function f, double arg, double a, double b, int n){
    double S1 = 0.0, S2 = 0.0, step = (b-a)/n;
    #pragma omp parallel for reduction(+:S1)
    for (int i = 1; i < n; i++){ if (i % 3 == 0) S1 += f(arg, a + i*step); }
    #pragma omp parallel for reduction(+:S2)
    for (int i = 1; i < n; i++){ if (i % 3 != 0) S2 += f(arg, a + i*step); }
    return 3./8. * (f(arg, a) + f(arg, b) + 2*S1 + 3*S2) * step;
}

double nc_method_xy(double a, double b, double c, double d, int n){
    double S1 = 0.0, S2 = 0.0, step = (b-a)/n, x, y;
    #pragma omp parallel for reduction(+:S1)
    for (int i = 1; i < n; i++){
        if (i % 3 == 0){
            x = a + i*step;
            S1 += nc_method_x(&g, x, c, d, n);
        }
    }
    #pragma omp parallel for reduction(+:S2)
    for (int i = 1; i < n; i++){
        if (i % 3 != 0){
            x = a + i*step;
            S2 += nc_method_x(&g, x, c, d, n);
        }
    }
    return 3./8. * (g(a, b) + g(c, d) + 2*S1 + 3*S2) * step;
}

double simpson_method_x(function f, double arg, double a, double b, int n){
    double S1 = 0.0, S2 = 0.0, step = (b-a)/n;
    #pragma omp parallel for reduction(+:S1)
    for (int i = 2; i < n; i+=2) S1 += f(arg, a + i*step);
    #pragma omp parallel for reduction(+:S2)
    for (int i = 1; i < n - 1; i+=2) S2 += f(arg, a + i*step);
    return (f(arg, a) + f(arg, b) + 2*S1 + 4*S2) * step/3.;
}

double simpson_method_xy(double a, double b, double c, double d, int n){
    double S1 = 0.0, S2 = 0.0, step = (b-a)/n, x,y;
    #pragma omp parallel for reduction(+:S1)
    for (int i = 2; i < n; i+=2){
        x = a + i*step;
        S1 += simpson_method_x(&g, x, c, d, n);
    }
    #pragma omp parallel for reduction(+:S2)
    for (int i = 1; i < n - 1; i+=2){
        x = a + i*step;
        S2 += simpson_method_x(&g, x, c, d, n);
    }
    return (g(a, b) + g(c, d) + 2*S1 + 4*S2) * step/3.;
}

double rect_method_x(function f, double arg, double a, double b, int n, int k){
    double S = 0.0, h = (b-a)/n;
    #pragma omp parallel for reduction(+:S)
    for (int i = k%2; i < n - k%3; i++) S += f(arg, a + i*h + (k%5)*h/2);
    return S * h;
}

double rect_method_xy(double a, double b, double c, double d, int n, int k){
    double S = 0.0, h = (b-a)/n; double x, y;
   #pragma omp parallel for reduction(+:S)
    for (int i = k%2; i < n - k%2; i++){
        x = a + i*h + (k%5)*h/2;
        S += rect_method_x(&g, x, c, d, n, k);
    }
    return S * h;
}

double trap_method_x(function f, double arg, double a, double b, int n){
    double S = 0.0, step = (b-a)/n;
    double diff = (f(arg, a) + f(arg, b)) / 2;
    #pragma omp parallel for reduction(+:S)
    for (int i = 1; i < n - 1; i++) S += f(arg, a + i*step);
    return (diff + S) * step;
}

double trap_method_xy(double a, double b, double c, double d, int n){
    double S = 0.0, step = (b-a)/n;
    double diff = (g(a,b) + g(c,d)) / 2 * (b-a), x, y;
    #pragma omp parallel for reduction(+:S)
    for (int i = 1; i < n - 1; i++){
        x = a + i*step;
        S += trap_method_x(&g, x, c, d, n);
    }
    return (diff + S) * step;
}
// I will use it in the future
void calc(double (*ptr)(function, double, double, double int), function f, double a, double b, double eps){
  clock_t start = clock();
  double res, delta = 1.0;
  int n = 1000, coef = (k == 0) ? 15. : 3.;
  while (eps < delta){
    res = (*ptr)(a,b,n,k);
    delta = (*ptr)(a,b,2*n,k) - res;
    n *= 2;
  }
  cout << res << " " << fabs(delta/coef) << " " <<(double)(clock() - start) << " ms" << endl;
}

// BROKEN ------->
int main(int argc, char* argv[]){
  omp_set_num_threads(4);
  double eps = 0.000001;
  double a = 0.4, b = 1.0;
  cout << "Result    " << "Delta       "  << "Time" << endl;
  //cout<< rect_method_x(&f, 0, a, b, 1000, 10) << endl;
  //cout<< rect_method_x(&f, 0, a, b, 1000, 15) << endl;
  //cout<< rect_method_x(&f, 0, a, b, 1000, 16) << endl;
  //cout<<trap_method_x(&f, 0, a, b, eps); << endl;
  //cout<< simpson_method_x(&f, 0, a, b, 1000) << endl;
  //cout<< nc_method_x(&f, 0, a, b, 1000) << endl;
  //cout<< mc_method_x(&f, 0, a, b, 0.0, 1.0, 1000) << endl;
  //cout<< rect_method_xy(1, 3, 0, 3, 1000, 10) << endl;
  //cout<< rect_method_xy(1, 3, 0, 3,1000, 15) << endl;
  //cout<< rect_method_xy(1, 3, 0, 3, 1000, 16) << endl;
  //cout<< trap_method_xy(1, 3, 0, 3, 1000) << endl;
  //cout<< simpson_method_xy(1, 3, 0, 3, 1000) << endl;
  //cout<< nc_method_xy(1, 3, 0, 3, 1000) << endl;
  //cout<< mc_method_xy(&g, 1, 3, 0, 3, 1000000) << endl;
  return 0;
}

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <string>
#include <chrono>
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

double mc_method_xy(function g, double a, double b, double c, double d, int n){
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

double nc_method_xy(function g, double a, double b, double c, double d, int n){
    double x, S1 = 0.0, S2 = 0.0, step = (b-a)/n;
    #pragma omp parallel for reduction(+:S1)
    for (int i = 1; i < n; i++){
        if (i % 3 == 0){
            x = a + i*step;
            S1 += nc_method_x(g, x, c, d, n);
        }
    }
    #pragma omp parallel for reduction(+:S2)
    for (int i = 1; i < n; i++){
        if (i % 3 != 0){
            x = a + i*step;
            S2 += nc_method_x(g, x, c, d, n);
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

double simpson_method_xy(function g, double a, double b, double c, double d, int n){
    double x, S1 = 0.0, S2 = 0.0, step = (b-a)/n;
    #pragma omp parallel for reduction(+:S1)
    for (int i = 2; i < n; i+=2){
        x = a + i*step;
        S1 += simpson_method_x(g, x, c, d, n);
    }
    #pragma omp parallel for reduction(+:S2)
    for (int i = 1; i < n - 1; i+=2){
        x = a + i*step;
        S2 += simpson_method_x(g, x, c, d, n);
    }
    return (g(a, b) + g(c, d) + 2*S1 + 4*S2) * step/3.;
}

double rect_method_x(function f, double arg, double a, double b, int n, int k){
    double S = 0.0, h = (b-a)/n;
    #pragma omp parallel for reduction(+:S)
    for (int i = k%2; i < n - k%3; i++) S += f(arg, a + i*h + (k%5)*h/2);
    return S * h;
}

double rect_method_xy(function g, double a, double b, double c, double d, int n, int k){
    double x, S = 0.0, h = (b-a)/n;
   #pragma omp parallel for reduction(+:S)
    for (int i = k%2; i < n - k%2; i++){
        x = a + i*h + (k%5)*h/2;
        S += rect_method_x(g, x, c, d, n, k);
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

double trap_method_xy(function g, double a, double b, double c, double d, int n){
    double x, S = 0.0, step = (b-a)/n;
    double diff = (g(a,b) + g(c,d)) / 2 * (b-a);
    #pragma omp parallel for reduction(+:S)
    for (int i = 1; i < n - 1; i++){
        x = a + i*step;
        S += trap_method_x(g, x, c, d, n);
    }
    return (diff + S) * step;
}

void calc(int method){
    double eps = 0.000001, res, delta = 1.0;
    int n = 10000, coef = 3;
    double f_a = 0.4, f_b = 1.0, f_c = 0.0, f_d = 1.0;
    double g_a = 1.0, g_b = 3.0, g_c = 0.0, g_d = 3.0;
    auto startTime = chrono::steady_clock::now();
    while (eps < delta){
        switch (method){
            case 1:{ res = rect_method_x(&f, 0, f_a, f_b, n, 10); delta = rect_method_x(&f, 0, f_a, f_b, 2*n, 10) - res; } break;
            case 2:{ res = rect_method_x(&f, 0, f_a, f_b, n, 15); delta = rect_method_x(&f, 0, f_a, f_b, 2*n, 15) - res; } break;
            case 3:{ res = rect_method_x(&f, 0, f_a, f_b, n, 16); delta = rect_method_x(&f, 0, f_a, f_b, 2*n, 16) - res; } break;
            case 4:{ res = trap_method_x(&f, 0, f_a, f_b, n); delta = trap_method_x(&f, 0, f_a, f_b, 2*n) - res; } break;
            case 5:{ res = simpson_method_x(&f, 0, f_a, f_b, n); delta = simpson_method_x(&f, 0, f_a, f_b, 2*n) - res; coef = 15; } break;
            case 6:{ res = nc_method_x(&f, 0, f_a, f_b, 3*n); delta = nc_method_x(&f, 0, f_a, f_b, 6*n) - res; } break;
            case 7:{ res = mc_method_x(&f, 0, f_a, f_b, f_c, f_d, n); delta = mc_method_x(&f, 0, f_a, f_b, f_c, f_d, 2*n) - res; } break;
            case 8:{ res = rect_method_xy(&g, g_a, g_b, g_c, g_d, n, 10); delta = rect_method_xy(&g, g_a, g_b, g_c, g_d, 2*n, 10) - res; } break;
            case 9:{ res = rect_method_xy(&g, g_a, g_b, g_c, g_d, n, 15); delta = rect_method_xy(&g, g_a, g_b, g_c, g_d, 2*n, 15) - res; } break;
            case 10:{ res = rect_method_xy(&g, g_a, g_b, g_c, g_d, n, 16); delta = rect_method_xy(&g, g_a, g_b, g_c, g_d, 2*n, 16) - res; } break;
            case 11:{ res = trap_method_xy(&g, g_a, g_b, g_c, g_d, n); delta = trap_method_xy(&g, g_a, g_b, g_c, g_d, 2*n) - res; } break;
            case 12:{ res = simpson_method_xy(&g, g_a, g_b, g_c, g_d, n); delta = simpson_method_xy(&g, g_a, g_b, g_c, g_d, 2*n) - res; coef = 15; } break;
            case 13:{ res = nc_method_xy(&g, g_a, g_b, g_c, g_d, 3*n); delta = nc_method_xy(&g, g_a, g_b, g_c, g_d, 6*n) - res; } break;
            case 14:{ res = mc_method_xy(&g, g_a, g_b, g_c, g_d, n); delta = mc_method_xy(&g, g_a, g_b, g_c, g_d, 2*n) - res; } break;
        }
    }
    auto endTime = chrono::steady_clock::now();
    auto runTime = chrono::duration_cast<std::chrono::duration<double>>(endTime-startTime);
    cout<< res << " " << fabs(delta/coef) << " " << runTime.count() << endl;
}

int main(int argc, char* argv[]){
    cout << "Result    " << "Delta       "  << "Time" << endl;
    for (int i = 1; i <= 4; i++){
        omp_set_num_threads(4);
        for (int j = 1; j <= 14; j++){
            calc(j);
        }
        cout<<"-----------"<<endl;
    }
    return 0;
}

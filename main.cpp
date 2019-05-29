#include <stdio.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <string>
#include <chrono>
#include <random>
#include <iomanip> 

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
    for (int i = k%2; i < n - k%3; i++){
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
    string names[7] = { "Left rect ", "Right rect ", "Middle rect ", "Trap ", "Simpson ", "3/8 ", "Monte Carlo " };
    double eps = 0.01, res, delta = 1.0, absdelta;
    double max_abs_dfx = 0.4621316416755651;
    double max_abs_ddfx = 1.3976907037089663;
    double max_abs_ddddfx = 13.522628329646611;
    int n = 1000, coef = 3, mk = 1000000;
    double f_a = 0.4, f_b = 1.0, f_c = 0.0, f_d = 1.0;
    double g_a = 1.0, g_b = 3.0, g_c = 0.0, g_d = 3.0;
    auto startTime = chrono::steady_clock::now();
    while (eps < delta){
        switch (method){
            case 0:{ res = rect_method_x(&f, 0, f_a, f_b, n, 10); delta = rect_method_x(&f, 0, f_a, f_b, 2*n, 10) - res; } break;
            case 1:{ res = rect_method_x(&f, 0, f_a, f_b, n, 15); delta = rect_method_x(&f, 0, f_a, f_b, 2*n, 15) - res; } break;
            case 2:{ res = rect_method_x(&f, 0, f_a, f_b, n, 16); delta = rect_method_x(&f, 0, f_a, f_b, 2*n, 16) - res; } break;
            case 3:{ res = trap_method_x(&f, 0, f_a, f_b, n); delta = trap_method_x(&f, 0, f_a, f_b, 2*n) - res; } break;
            case 4:{ res = simpson_method_x(&f, 0, f_a, f_b, n); delta = simpson_method_x(&f, 0, f_a, f_b, 2*n) - res; coef = 15; } break;
            case 5:{ res = nc_method_x(&f, 0, f_a, f_b, 3*n); delta = nc_method_x(&f, 0, f_a, f_b, 6*n) - res; } break;
            case 6:{ res = mc_method_x(&f, 0, f_a, f_b, f_c, f_d, mk); delta = mc_method_x(&f, 0, f_a, f_b, f_c, f_d, 2*mk) - res; } break;
            case 7:{ res = rect_method_xy(&g, g_a, g_b, g_c, g_d, n, 10); delta = rect_method_xy(&g, g_a, g_b, g_c, g_d, 2*n, 10) - res; } break;
            case 8:{ res = rect_method_xy(&g, g_a, g_b, g_c, g_d, n, 15); delta = rect_method_xy(&g, g_a, g_b, g_c, g_d, 2*n, 15) - res; } break;
            case 9:{ res = rect_method_xy(&g, g_a, g_b, g_c, g_d, n, 16); delta = rect_method_xy(&g, g_a, g_b, g_c, g_d, 2*n, 16) - res; } break;
            case 10:{ res = trap_method_xy(&g, g_a, g_b, g_c, g_d, n); delta = trap_method_xy(&g, g_a, g_b, g_c, g_d, 2*n) - res; } break;
            case 11:{ res = simpson_method_xy(&g, g_a, g_b, g_c, g_d, n); delta = simpson_method_xy(&g, g_a, g_b, g_c, g_d, 2*n) - res; coef = 15; } break;
            case 12:{ res = nc_method_xy(&g, g_a, g_b, g_c, g_d, 3*n); delta = nc_method_xy(&g, g_a, g_b, g_c, g_d, 6*n) - res; } break;
            case 13:{ res = mc_method_xy(&g, g_a, g_b, g_c, g_d, mk); delta = mc_method_xy(&g, g_a, g_b, g_c, g_d, 2*mk) - res; } break;
        }
        n *= 2;
    }
    auto endTime = chrono::steady_clock::now();
    auto runTime = chrono::duration_cast<std::chrono::duration<double>>(endTime-startTime);
        switch (method){
            case 0:{ absdelta = max_abs_dfx*(pow(f_b-f_a,2)/2); } break;
            case 1:{ absdelta = max_abs_dfx*(pow(f_b-f_a,2)/2); } break;
            case 2:{ absdelta = max_abs_ddfx*(pow(f_b-f_a,3)/24); } break;
            case 3:{ absdelta = max_abs_ddfx*(pow(f_b-f_a,3)/12); } break;
            case 4:{ absdelta = max_abs_ddddfx*(pow(f_b-f_a,5)/2880); } break;
            case 5:{ absdelta = max_abs_ddddfx*(pow(f_b-f_a,5)/2880); } break;
            case 6:{ absdelta = 3*(((f_b-f_a)/(2*sqrt(3)))/sqrt(n) + ((f_d-f_c)/(2*sqrt(3)))/sqrt(n))/2; } break; // Согласно равномерному закону распредления
            case 7:{  } break;
            case 8:{  } break;
            case 9:{ } break;
            case 10:{  } break;
            case 11:{ } break;
            case 12:{  } break;
            case 13:{  } break;
        }
    cout<< setw(12) <<names[method%7] << setw(8) << res << " " << setw(11) << fabs(delta/coef) << " " << setw(11) << absdelta << " " << setw(11) << runTime.count() << endl;
}

int main(int argc, char* argv[]){
    cout << setw(12) << "Method " << setw(8) << "Result" << " " << setw(11) << "Delta" << " " << setw(11) << "Absdelta"  << " " << setw(11) << "Time" << endl;
    for (int i = 1; i <= 4; i++){
        omp_set_num_threads(i);
        for (int j = 0; j <= 13; j++){
            calc(j);
        }
        cout<<"-----------"<<endl;
    }
    return 0;
}

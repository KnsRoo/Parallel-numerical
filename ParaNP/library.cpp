#include "library.h"

void randdouble(double min, double max, double *arr, int n){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        arr[i] = dis(gen);
    }
}

double g(double x, double y){
    return  ((x >= 0) and (x < 2*y)) ? pow(y,2)*exp(-x*y/8) : 0;
}

double k(double arg, double t){
  return (19*pow(t,7)+84*pow(t,4)+35)*exp(-2.4*t);
}

double f(double arg, double t){
  return (27/(39*t+16))+7;
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

double calculate_xy(int method, int parts){
    double LOWER_LIMIT = 0.0, UPPER_LIMIT = 4.0,
             DOWN_LINE = 0.0,     UP_LINE = 2.0;
    double result = 0;
    switch (method) {
        case 0: result = rect_method_xy(g,LOWER_LIMIT,UPPER_LIMIT,DOWN_LINE,UP_LINE,parts,10); break;
        case 1: result = rect_method_xy(g,LOWER_LIMIT,UPPER_LIMIT,DOWN_LINE,UP_LINE,parts,15); break;
        case 2: result = rect_method_xy(g,LOWER_LIMIT,UPPER_LIMIT,DOWN_LINE,UP_LINE,parts,16); break;
        case 3: result = trap_method_xy(g,LOWER_LIMIT,UPPER_LIMIT,DOWN_LINE,UP_LINE,parts); break;
        case 4: result = simpson_method_xy(g,LOWER_LIMIT,UPPER_LIMIT,DOWN_LINE,UP_LINE,parts); break;
        case 5: result = mc_method_xy(g,LOWER_LIMIT,UPPER_LIMIT,DOWN_LINE,UP_LINE,parts); break;
    default: result = 0.0; break;
    }
    return result;
}

double calculate(int method, int parts, bool first){
    double LOWER_LIMIT, UPPER_LIMIT, DOWN_LINE,UP_LINE;
    function func;
    if (first){
        LOWER_LIMIT = 0.0, UPPER_LIMIT = 8.0,
        DOWN_LINE   = 0.0, UP_LINE     = 5.0;
        func = &f;
    } else {
        LOWER_LIMIT = 0.0, UPPER_LIMIT = 15.0,
        DOWN_LINE   = 0.0, UP_LINE     = 10.0;
        func = &k;
    }
    double result = 0;
    switch (method) {
        case 0: result = rect_method_x(func,0,LOWER_LIMIT,UPPER_LIMIT,parts,10); break;
        case 1: result = rect_method_x(func,0,LOWER_LIMIT,UPPER_LIMIT,parts,15); break;
        case 2: result = rect_method_x(func,0,LOWER_LIMIT,UPPER_LIMIT,parts,16); break;
        case 3: result = trap_method_x(func,0,LOWER_LIMIT,UPPER_LIMIT,parts); break;
        case 4: result = simpson_method_x(func,0,LOWER_LIMIT,UPPER_LIMIT,parts); break;
        case 5: result = mc_method_x(func,0,LOWER_LIMIT,UPPER_LIMIT,DOWN_LINE,UP_LINE,parts); break;
    default: result = 0.0; break;
    }
    return result;
}

double inaccuracy_xy(int method, int parts){
    double LOWER_LIMIT = 0.0, UPPER_LIMIT = 4.0,
             DOWN_LINE = 0.0,     UP_LINE = 2.0;
    double dfx = 1.0, dfy = 4.0,
           d2fx = 0.25, d2fy = 2.0,
           d4fx = 0.015625, d4fy = 3.0;
    double result = 0.0;
    switch (method) {
        case 0: result = dfx*(pow(UPPER_LIMIT-LOWER_LIMIT,2))/(2*parts)+dfy*(pow(UP_LINE-DOWN_LINE,2))/(2*parts); break;
        case 1: result = dfx*(pow(UPPER_LIMIT-LOWER_LIMIT,2))/(2*parts)+dfy*(pow(UP_LINE-DOWN_LINE,2))/(2*parts); break;
        case 2: result = d2fx*(pow(UPPER_LIMIT-LOWER_LIMIT,3))/(24*pow(parts,2))+d2fy*(pow(UP_LINE-DOWN_LINE,3))/(24*pow(parts,2)); break;
        case 3: result = d2fx*(pow(UPPER_LIMIT-LOWER_LIMIT,3))/(12*pow(parts,2))+d2fy*(pow(UP_LINE-DOWN_LINE,3))/(12*pow(parts,2)); break;
        case 4: result = (d4fx*(pow(UPPER_LIMIT-LOWER_LIMIT,5)))/(2880*pow(parts,4))+(d4fy*(pow(UP_LINE-DOWN_LINE,5)))/(2880*pow(parts,4)); break;
        case 5: result = 3*sqrt((pow((UPPER_LIMIT-LOWER_LIMIT),2)/12*parts); break;
    default: break;
    }
    return result;
}

double inaccuracy(int method, int parts, bool first){
    double LOWER_LIMIT, UPPER_LIMIT, DOWN_LINE,UP_LINE;
    double dfx = 0.0, d2fx = 0.0, d4fx = 0.0;
    if (first){
        LOWER_LIMIT = 0.0, UPPER_LIMIT = 8.0,
        DOWN_LINE   = 0.0, UP_LINE     = 5.0;
        dfx = 4.11328125, d2fx = 20.05224609375, d4fx = 1429.662483215332;
    } else {
        LOWER_LIMIT = 0.0, UPPER_LIMIT = 15.0,
        DOWN_LINE   = 0.0, UP_LINE     = 10.0;
        dfx = 84.0, d2fx = 201.6, d4fx = 3177.216;
    }
    double result = 0;
    switch (method) {
        case 0: result = dfx*(pow(UPPER_LIMIT-LOWER_LIMIT,2))/(2*parts); break;
        case 1: result = dfx*(pow(UPPER_LIMIT-LOWER_LIMIT,2))/(2*parts); break;
        case 2: result = d2fx*(pow(UPPER_LIMIT-LOWER_LIMIT,3))/(24*pow(parts,2)); break;
        case 3: result = d2fx*(pow(UPPER_LIMIT-LOWER_LIMIT,3))/(12*pow(parts,2)); break;
        case 4: result = d4fx*(pow(UPPER_LIMIT-LOWER_LIMIT,5))/(2880*pow(parts,4)); break;
        case 5: result = 3*sqrt((pow((UPPER_LIMIT-LOWER_LIMIT),2)/12*parts); break;
    default: result = 0.0; break;
    }
    return result;
}

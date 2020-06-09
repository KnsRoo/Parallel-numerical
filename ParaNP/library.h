#ifndef LIBRARY_H
#define LIBRARY_H
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <string>
#include <chrono>
#include <random>
#include <iomanip>

typedef double(*function)(double, double);

void randdouble(double min, double max, double *arr, int n);

double f(double arg, double t);

double k(double arg, double t);

double g(double x, double y);

double mc_method_x(function f,double a, double b, double c, double d, int n);

double simpson_method_x(function f,double a, double b, int n);

double rect_method_x(function f,double a, double b, int n, int k);

double trap_method_x(function f,double a, double b, int n);

double mc_method_xy(function g,double a, double b, double c, double d, int n);

double simpson_method_xy(function g,double a, double b, double c, double d, int n);

double rect_method_xy(function g,double a, double b, double c, double d, int n, int k);

double trap_method_xy(function g,double a, double b, double c, double d, int n);

double calculate(int method, int parts, bool first);

double calculate_xy(int method, int parts);

double inaccuracy(int method, int parts, bool first);

double inaccuracy_xy(int method, int parts);

#endif // LIBRARY_H

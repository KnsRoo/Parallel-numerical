import sympy
from sympy.utilities.lambdify import lambdify
import scipy
import scipy.optimize

x = sympy.Symbol('x')
f = (pow (x, 2) + sympy.sin(0.48 * (x + 2))) / (sympy.exp(pow (x, 2)) + 0.38)

df = f.diff(x)
ddf = df.diff(x)
dddf = ddf.diff(x)
ddddf = dddf.diff(x)

df = -abs(df)
ddf = -abs(ddf)
dddf = -abs(dddf)
ddddf = -abs(ddddf)

df = lambdify(x, df, 'scipy')
ddf = lambdify(x, ddf, 'scipy')
dddf = lambdify(x, dddf, 'scipy')
ddddf = lambdify(x, ddddf, 'scipy')

x1 = scipy.optimize.fminbound(df, 0.4, 1)
x2 = scipy.optimize.fminbound(ddf, 0.4, 1)
x3 = scipy.optimize.fminbound(dddf, 0.4, 1)
x4 = scipy.optimize.fminbound(ddddf, 0.4, 1)

print("x_max", x1, "max abs df(x_max)", -df(x1))
print("x_max", x2, "max abs ddf(x_max)", -ddf(x2))
print("x_max", x3, "max abs dddf(x_max)", -dddf(x3))
print("x_max", x4, "max abs ddddf(x_max)", -ddddf(x4))

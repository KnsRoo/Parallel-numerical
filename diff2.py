import sympy, numpy as np

x = sympy.Symbol('x')
y = sympy.Symbol('y')
f = (x + y)**2/x

# d4/x^2*y^2
dfx = f.diff(x)
dfx = dfx.diff(x)
df = dfx.diff(y)
df = df.diff(y)

#d2/y^2
dfy = f.diff(y)
dfy = df.diff(y)

Xs = []

def g(a,b):
  return abs(-2/a**2 + 2*(a+b)/a**3)

print(df, dfy, dfx)

xi = list(np.linspace(1,3,1000))
yi = list(np.linspace(0,3,1300))

for i in range(len(xi)):
  Xs.append(g(xi[i],yi[i]))

print(max(Xs))

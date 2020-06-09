import sympy, numpy as np, math

x = sympy.Symbol('x')
y = sympy.Symbol('y')
f = y**2*pow(math.e,-x*y/8)
g1 = (27/(39*x+16))+7
g2 = (19*pow(x,7)+84*pow(x,4)+35)*pow(math.e,-2.4*x)

g1_dfx = g1.diff(x)
g1_dfxx = g1.diff(x,2)
g1_dfxxxx = g1.diff(x,4)

g2_dfx = g2.diff(x)
g2_dfxx = g2.diff(x,2)
g2_dfxxxx = g2.diff(x,4)

func_g1_dfx = sympy.lambdify(x,g1_dfx)
func_g1_dfxx = sympy.lambdify(x,g1_dfxx)
func_g1_dfxxxx = sympy.lambdify(x,g1_dfxxxx)

func_g2_dfx = sympy.lambdify(x,g2_dfx)
func_g2_dfxx = sympy.lambdify(x,g2_dfxx)
func_g2_dfxxxx = sympy.lambdify(x,g2_dfxxxx)

s1 = []
s2 = []
s3 = []
s4 = []
s5 = []
s6 = []

xi1 = list(np.linspace(0,8,15))
xi2 = list(np.linspace(0,15,15))

for i in range(len(xi1)):
	s1.append(abs(func_g1_dfx(xi1[i])))
	s2.append(abs(func_g1_dfxx(xi1[i])))
	s3.append(abs(func_g1_dfxxxx(xi1[i])))
	s4.append(abs(func_g2_dfx(xi2[i],)))
	s5.append(abs(func_g2_dfxx(xi2[i])))
	s6.append(abs(func_g2_dfxxxx(xi2[i])))
print(g2_dfx)
print("g1_dfx",max(s1))
print("g1_dfxx",max(s2))
print("g1_dfxxxx",max(s3))
print("g2_dfx",max(s4))
print("g2_dfxx",max(s5))
print("g2_dfxxxx",max(s6))

print("-"*10)

# d4/x^2*y^2
dfx = f.diff(x)
dfy = f.diff(y)

dfxx = dfx.diff(x)
dfyy = dfy.diff(y)
dfxy = dfx.diff(y)

dfxxx = dfxx.diff(x)
dfyyy = dfyy.diff(y)

dfxxxx = dfxx.diff(x,2)
dfyyyy = dfyy.diff(y,2)
dfxxyy = dfxx.diff(y,2)
dfxxxy = dfxxx.diff(y)
dfyyyx = dfyyy.diff(x)


l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
l7 = []
l8 = []
l9 = []
l10 = []


#first derivative
func_dfx = sympy.lambdify([x,y],dfx)
func_dfy = sympy.lambdify([x,y],dfy)
#second derivative
func_dfxx = sympy.lambdify([x,y],dfxx)
func_dfyy = sympy.lambdify([x,y],dfyy)
func_dfxy = sympy.lambdify([x,y],dfxy)
#fourth derivative
func_dfxxxx = sympy.lambdify([x,y],dfxxxx)
func_dfyyyy = sympy.lambdify([x,y],dfyyyy)
func_dfxxyy = sympy.lambdify([x,y],dfxxyy)
func_dfxxxy = sympy.lambdify([x,y],dfxxxy)
func_dfyyyx = sympy.lambdify([x,y],dfyyyx)


xi = list(np.linspace(0,4,100))
yi = list(np.linspace(0,2,100))

for i in range(len(xi)):
	for j in range(len(yi)):
		print("i,j",i,j)
		l1.append(abs(func_dfx(xi[i],yi[j])))
		l2.append(abs(func_dfy(xi[i],yi[j])))
		l3.append(abs(func_dfxx(xi[i],yi[j])))
		l4.append(abs(func_dfyy(xi[i],yi[j])))
		#l5.append(abs(func_dfxy(xi[i],yi[j])))
		l6.append(abs(func_dfxxxx(xi[i],yi[j])))
		l7.append(abs(func_dfyyyy(xi[i],yi[j])))
		#l8.append(abs(func_dfxxyy(xi[i],yi[j])))
		#l9.append(abs(func_dfxxxy(xi[i],yi[j])))
		#l10.append(abs(func_dfyyyx(xi[i],yi[j])))

print("dfx",max(l1))
print("dfy",max(l2))
print("dfxx",max(l3))
print("dfyy",max(l4))
#print("dfxy",max(l5))
print("dfxxxx",max(l6))
print("dfyyyy",max(l7))
#print("dfxxyy",max(l8))
#print("dfxxxy",max(l9))
#print("dfyyyx",max(l10))

a,b,c,d = 0,4,0,2
parts = 2
print("df",max(l1)*(pow(b-a,2))/(2*parts))
print("d2f",max(l3)*(pow(b-a,3))/(24*pow(parts,2)))
print("d4f",max(l6)*(pow(b-a,5))/(2880*pow(parts,4)))

print("df",max(l1)*(pow(b-a,2))/(2*parts)+max(l2)*(pow(d-c,2))/(2*parts))
print("d2f",max(l3)*(pow(b-a,3))/(24*pow(parts,2))+max(l4)*(pow(d-c,3))/(24*pow(parts,2)))
print("d4f",max(l6)*(pow(b-a,5))/(2880*pow(parts,4))+max(l7)*(pow(d-c,5))/(2880*pow(parts,4)))


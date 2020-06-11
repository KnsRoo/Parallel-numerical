import sympy, numpy as np, math

def derr_s(g, interval, keys):
	y_ex = 'y' in str(g)
	s = []
	x = sympy.Symbol('x')
	y = sympy.Symbol('y')
	if y_ex:
		funcs = [sympy.lambdify([x,y],g.diff(x,k)) for k in [1,2,4]]
		funcs.extend([sympy.lambdify([x,y],g.diff(y,k)) for k in [1,2,4]])
	else:
		funcs = [sympy.lambdify(x,g.diff(x,k)) for k in [1,2,4]]
	a,b,c,d = interval if y_ex else (*interval,0,0)
	xi, yi = list(np.linspace(a,b,100)),list(np.linspace(c,d,100))
	for func in funcs:
		tmp = []
		if y_ex:
			for x in xi:
				for y in yi:
					tmp.append(abs(func(x,y)))
		else:
			for x in xi:
				tmp.append(abs(func(x)))
		s.append(tmp)
	keys = keys if y_ex else keys[:3]
	for keys,values in dict(zip(keys,[max(s[i]) for i in range(len(s))])).items():
			print(keys, values)

if __name__ == '__main__':
	x = sympy.Symbol('x')
	y = sympy.Symbol('y')
	keys = ["dfx:","d2fx:","d4fx:","dfy:","d2fy:","d4fy:"]
	gs = [(27/(39*x+16))+7, (19*pow(x,7)+84*pow(x,4)+35)*pow(math.e,-2.4*x),y**2*pow(math.e,-x*y/8)]
	intervals = [(0,8),(0,15),(0,4,0,2)]
	for item in gs:
		print("f:",item)
		derr_s(item,intervals[gs.index(item)],keys)

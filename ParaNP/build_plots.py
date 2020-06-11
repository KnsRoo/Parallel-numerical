import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def build_plots(data, max_y_time, name, max_y_sp = 4, max_y_ep = 2):
	titles = {"Left rect":   "Метод левых прямоугольников",
			  "Right rect":	 "Метод правых прямоугольников",
			  "Middle rect": "Метод средних прямоугольников",
			  "Trap":	     "Метод Трапеций",
			  "Simpson":	 "Метод Симпсона",
			  "Monte Carlo": "Метод Монте-Карло"}
	for method in sorted(set(data['Method'])):
		sub_data = data.loc[data['Method'] == method]
		fig = plt.figure(figsize=(8, 6), dpi=100)

		rt_subplt = fig.add_subplot(211)
		Sp_subplt = fig.add_subplot(223)
		Ep_subplt = fig.add_subplot(224)
		rt_subplt.set_title(titles[method])
		rt_subplt.set_xlabel('Число потоков')
		rt_subplt.set_ylabel('Время выполнения')
		Sp_subplt.set_title('Ускорение')
		Sp_subplt.set_xlabel('Число потоков')
		Sp_subplt.set_ylabel("$S_{p}$")
		Ep_subplt.set_title('Эффективность')
		Ep_subplt.set_xlabel('Число потоков')
		Ep_subplt.set_ylabel("$E_{p}$")

		for dimension in sorted(set(sub_data['Dimension'])):
		    sub_df = sub_data.loc[data.Dimension == dimension]
		    one_thread_t = sub_df[sub_df.NumThreads == 1]['RunTime'].astype(float)
		    speedup = np.array(one_thread_t) / np.array(sub_df['RunTime'])
		    efficiency = speedup / np.array(sub_df['NumThreads'])
		    rt_subplt.plot(sub_df['NumThreads'], sub_df['RunTime'],
		                       marker="o", label="{dim}".format(dim=dimension))
		    Sp_subplt.plot(sub_df['NumThreads'], speedup, marker=".",
		                       label="{dim}".format(dim=dimension))
		    Ep_subplt.plot(sub_df['NumThreads'], efficiency,
		                       marker=".", label="{dim}".format(dim=dimension))
		rt_subplt.legend(loc = "upper right")
		plt.subplots_adjust(wspace=0.5, hspace=0.6)
		fig.savefig(name + method + '.png')



if __name__ == '__main__':
	data_1, data_2 = pd.read_csv('result1.csv'), pd.read_csv('result2.csv')
	max_t1,max_t2 = max(data_1['RunTime']),max(data_2['RunTime'])
	build_plots(data_1,max_t1, "1_")
	build_plots(data_2,max_t2, "2_")

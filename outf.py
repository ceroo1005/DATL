import os
import seaborn as sns
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.axes_grid1 import host_subplot

x_major_locator = MultipleLocator(5)
pattern = re.compile(r'\d+')

rootpath = 'results/d-w'
result_text = []
for _ in os.listdir(rootpath):
	if _.split('.')[-1] == 'txt':
		result_text.append(_)

result_text.sort()
result = []
for name in result_text:
	txtpath = os.path.join(rootpath, name)
	fo = open(txtpath, "r")

	ref = re.findall(r'\d\.\d+|\d+',fo.readlines()[-1])[1:]
	result.append([int(name[:-4].split('-')[-1])] + list(map(lambda x:float(x), ref)))
	fo.close()

result.sort(key=lambda x: x[0])

result = np.array(result).T

baz = list(map(lambda x:100*x, result[1]))
Openness = result[0]

for i,j,k in zip(Openness, result[-1], baz):
	print(i,j,k)
# # Openness = list(map(lambda x:str(100*x/3).split('.')[0]+'%', result[0]))
# g = sns.relplot(x=Openness, y=result[-1], size=baz, alpha=0.7,
#             sizes=(20, 250))
# # leg = g._legend
# # leg.set_bbox_to_anchor([1,0.8])
# # leg.columnspacing = 10
# g._legend.remove()
# # leg._loc = 10

# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.legend(borderaxespad=1)

# plt.ylabel('Accuracy')
# plt.xlabel('Openness')
# plt.xticks(rotation=-60)
# plt.tight_layout()

def Openness(ct, cs=3):
	return 1 - cs/ct

# plt.show()
fig, ax = plt.subplots()
ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


intbaz = [int(_) for _ in baz]
for _, vbaz in enumerate(set(intbaz)):
	lindex = np.where(np.array(intbaz) == vbaz)
	ax.scatter(Openness(ct=result[0][lindex]), result[-1][lindex], s=np.exp(result[1][lindex]*32), alpha=0.5, label=str(int(vbaz))+' %')

ax.set_xlabel(r'Openness', fontsize=12)
ax.set_ylabel(r'Accuracy', fontsize=12)
# ax.set_title('Volume and percent change')
ax.legend()
fig.tight_layout()

plt.savefig(f"results/d-w/Open.pdf", dpi=150)
plt.show()




host = host_subplot(111)
par = host.twinx()

host.set_xlabel("Openness")
host.set_ylabel("Accuracy")
par.set_ylabel("Sampling Proportion")

p1, = host.plot(Openness(result[0]), result[-1], label="Accuracy")
p2, = par.plot(Openness(result[0]), result[1], label="Sampling Proportion")

host.set_ylim(0.8, 1)
par.set_ylim(0, 0.3)
leg = plt.legend()

host.yaxis.get_label().set_color(p1.get_color())
leg.texts[0].set_color(p1.get_color())

par.yaxis.get_label().set_color(p2.get_color())
leg.texts[1].set_color(p2.get_color())

plt.savefig(f"results/d-w/OpenLine.pdf", dpi=150)
plt.show()



import matplotlib.pyplot as plt
import numpy as np

labels = ['3-10', '3-15']
labels = [ 'LoopNet-DSAN', 'LoopNet-DANN', '*DSAN', '*DANN']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Men')
rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()

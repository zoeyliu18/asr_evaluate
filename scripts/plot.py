import matplotlib.pyplot as plt
import sys
import re
import numpy as np
from matplotlib.lines import Line2D


f = open(sys.argv[1])
randoms = []
people = []
feats = []
dista = []
featlabels = []
f.readline()
for line in f:
    parts = line.rstrip().split("\t")
    if parts[-1] == "random_different":
        randoms.append(float(parts[3]))
    elif parts[-1] == "heldout_speaker":
        people.append(float(parts[3]))
    elif parts[-1] == "distance":
        dista.append(float(parts[3]))
    else:
        feats.append(float(parts[3]))
f.close()

markies = ["x", "+", "^", "<", ">", "v"]
colories = ["red", "purple", "brown", "blue", "green", "orange"]
featlabels = ["duration", "pitch", "intensity", "perplexity", "#words", "#tokens"]  

fig, ax = plt.subplots()
plt.boxplot([people, randoms, dista], patch_artist=True, boxprops=dict(facecolor="blue"), flierprops={'marker': '.', 'markersize':4})
plt.plot ([1]*len(people), people, '.', color="blue", markersize=4)
plt.plot([1], np.mean(people), '.', color='black', markersize=10)
plt.plot ([2]*len(randoms), randoms, '.', color="grey", markersize=4)
plt.plot([2], np.mean(randoms), '.', color='black', markersize=10)
plt.plot ([3]*len(dista), dista, '.', color="pink", markersize=4)
plt.plot([3], np.mean(dista), '.', color='black', markersize=10)
for j in range(len(feats)):
    plt.plot([4], feats[j], markies[j], color=colories[j])

ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(["speaker", "random", "adversarial", "heuristic"])
ax.set_ylabel("WER")


legend_elements = []
for i in range(len(markies)):
    le =  Line2D([0], [0], marker=markies[i], color=colories[i], label=featlabels[i], markerfacecolor=colories[i], markersize=4, linestyle='')
    legend_elements.append(le)

ax.legend(handles=legend_elements)

plt.show()

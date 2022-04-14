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

    # next two lines do not apply to hupa
    elif parts[-1] in ["heldout_speaker", "heldout_session"]:
        people.append(float(parts[3]))

    elif parts[-1] == "distance":
        dista.append(float(parts[3]))

    else:
        # this does not apply to hupa
        feats.append(float(parts[3]))

        # this applies to hupa
        #people.append(float(parts[3]))
f.close()

markies = ["x", "+", "^", "<", ">", "v"]
colories = ["red", "purple", "brown", "blue", "green", "orange"]
featlabels = ["duration", "pitch", "intensity", "perplexity", "#words", "#tokens"]  

fig, ax = plt.subplots()
plt.boxplot([people, randoms, dista], flierprops={'marker': '.', 'markersize':4}, patch_artist=True, showmeans=True, medianprops={"color":"black"},  meanprops={"marker":".","markersize":10,"markerfacecolor":"black", "markeredgecolor":"black"})
#plt.plot ([1]*len(people), people, '.', color="blue", markersize=4)
#plt.plot([1], np.mean(people), '.', color='black', markersize=10)
#plt.plot ([2]*len(randoms), randoms, '.', color="grey", markersize=4)
#plt.plot([2], np.mean(randoms), '.', color='black', markersize=10)
#plt.plot ([3]*len(dista), dista, '.', color="pink", markersize=4)
#plt.plot([3], np.mean(dista), '.', color='black', markersize=10)

for j in range(len(feats)):
    plt.plot([4], feats[j], markies[j], color=colories[j])

ax.set_xticks([1, 2, 3, 4])

ax.set_xticklabels(["speaker", "random", "adversarial", "heuristic"])

if 'hupa' in sys.argv[1] or 'swahili' in sys.argv[1]:
    ax.set_xticklabels(["session", "random", "adversarial", "heuristic"])


ax.set_ylabel("WER", size=14)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylim(ymin=0, ymax=100)

ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)


legend_elements = []
for i in range(len(markies)):
    le =  Line2D([0], [0], marker=markies[i], color=colories[i], label=featlabels[i], markerfacecolor=colories[i], markersize=4, linestyle='')
    legend_elements.append(le)

# The line below is for making legends in the graph
ax.legend(handles=legend_elements, prop={'size': 10})

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

figsave = re.sub("_.*$", ".png", sys.argv[1])

fig.set_size_inches(6, 3)
fig.savefig(figsave, dpi=300)


plt.show()



#plt.plot ([1]*len(people), people, '.', color="black", markersize=4)
#plt.plot ([2]*len(randoms), randoms, '.', color="black", markersize=4)
#plt.plot ([3]*len(dista), dista, '.', color="black", markersize=4)

#plt.plot([1], np.mean(people), '.', color='black', markersize=10)
#plt.plot([2], np.mean(randoms), '.', color='black', markersize=10)
#plt.plot([3], np.mean(dista), '.', color='black', markersize=10)


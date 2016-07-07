import matplotlib.pyplot as plt
import numpy as np


filename = '../evaluation/plot_data/coco-PR-fcn.txt'
heatmap = np.loadtxt(filename)

filename = '../evaluation/plot_data/coco-PR-TP-FULL.txt'
textProposals = np.loadtxt(filename)

filename = '../evaluation/plot_data/coco-PR-HM_MAX.txt'
half = np.loadtxt(filename)


plt.plot(heatmap[:,0], heatmap[:,1], 'b', label = 'fcn')
plt.plot(textProposals[:,0], textProposals[:,1], 'r', label = 'textProposals')
plt.plot(half[:,0], half[:,1], 'g', label = 'heatmap max')

plt.axis([0, 1, 0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Image containing text classification')
plt.legend(loc='upper right')
plt.show()

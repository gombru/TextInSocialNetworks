import matplotlib.pyplot as plt
import numpy as np

filename = '../evaluation/plot_data_BU_0.7/coco-OP-heatmap-0.7.txt'
heatmap = np.loadtxt(filename)

filename = '../evaluation/plot_data_BU_0.7/coco-OP-textProposals-0.7.txt'
textProposals = np.loadtxt(filename)

filename = '../evaluation/plot_data_BU_0.7/coco-OP-mean-0.7.txt'
half = np.loadtxt(filename)


plt.plot(heatmap[:,0], heatmap[:,1], 'b', label = 'fcn')
plt.plot(textProposals[:,0], textProposals[:,1], 'r', label = 'textProposals')
plt.plot(half[:,0], half[:,1], 'g', label = 'fusion')

plt.ylim(0, 1)
plt.xscale('log')
plt.xlim(1, 10000)
plt.title('IoU = 0.7')
plt.xlabel('# of proposals')
plt.ylabel('Detection rate')
plt.legend(loc='upper right')
plt.show()
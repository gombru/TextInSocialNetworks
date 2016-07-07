import glob, os
import numpy
import json
import sys
import matplotlib.pyplot as plt

sys.path.append('coco-text-api')
import coco_evaluation
import coco_text
import numpy as np

filename = '../../data/coco-text/images_evaluation/hm_max.txt'
out_filename = 'plot_data/coco-PR-HM_MAX.txt'
#score_thresholds = np.arange(0.96002,1.0002,0.0002)
score_thresholds = np.arange(0,1.001,0.005)

#Read annotations
ct = coco_text.COCO_Text('../../data/coco/coco-text-annotations/COCO_Text.json')

# Read images data
read = numpy.loadtxt(filename, delimiter=" ")
data = np.zeros((len(read[:,0]),len(read[0,:]) + 1))
data[:,:-1] = read

# Retrieve images with legible
imgIds = ct.getImgIds(imgIds=ct.val,
                       catIds=[('legibility', 'legible')])

# Check if image contains text or not
for i in range(0, len(data[:,0])):
    if (data[i,0] in imgIds):
        data[i, 2] = 1


precision = np.ones((score_thresholds.__len__()))
recall = np.zeros((score_thresholds.__len__()))

for s in range(0,score_thresholds.__len__()):

    print 'Evaluating for score threshold ' + str(score_thresholds[s]) + ' ...'
    tp = 0; fp = 0; fn = 0;

    for i in range(0, len(data[:, 0])):
        if (data[i, 2] == 1):
            if(data[i, 1] > score_thresholds[s]):
                tp += 1
            else:
                fn += 1
        elif (data[i, 1] > score_thresholds[s]):
            fp += 1


    #Compute precision and recall
    if(tp + fp > 0):
        precision[s] = float(tp) / (tp + fp)
    else:
        precision[s] = 1

    if (fn + tp > 0):
        recall[s] = float(tp) / (fn + tp)
    #print fp
    print 'Evaluation computed'


recall = recall[::-1]
precision = precision[::-1]

# Save evaluation data
np.savetxt(out_filename, np.c_[recall,precision])

#Plot
plt.plot(recall, precision)
plt.axis([0, 1, 0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Image containing text classification')
plt.show()

print 'Done'
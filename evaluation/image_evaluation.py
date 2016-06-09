import glob, os
import numpy
import json
import sys
import matplotlib.pyplot as plt

sys.path.append('coco-text-api')
import coco_evaluation
import coco_text
import numpy as np

ct = coco_text.COCO_Text('../../data/coco/coco-text-annotations/COCO_Text.json')

# Retrieve images with legible
imgIds = ct.getImgIds(imgIds=ct.val,
                       catIds=[('legibility', 'legible')])

score_thresholds = np.arange(0,1.1,0.01)
precision = np.zeros((score_thresholds.__len__()))
recall = np.zeros((score_thresholds.__len__()))

for s in range(0,score_thresholds.__len__()):
    positive_images = []
    negative_images = []

    #Classify images with scores
    for file in glob.glob("../../data/coco-text/images_evaluation/*.txt"):
        data = numpy.loadtxt(file,delimiter=" ")
        for r in range (0,len(data)):
            #If image score is above threshold, is positive
            if(data[r,1] > score_thresholds[s]):
                positive_images.append(int(data[r,0]))
            else:
                negative_images.append(int(data[r, 0]))

    # Evaluate
    tp = 0; fp = 0; fn = 0;
    for i in range(0, len(positive_images)):
        if positive_images[i] in imgIds:
            tp += 1
        else:
            fp += 1

    for i in range(0,len(negative_images)):
        if negative_images[i] in imgIds:
            fn += 1

    #Compute precision and recall
    if(tp + fp > 0):
        precision[s] = float(tp) / (tp + fp)
    else:
        precision[s] = 1

    if (fn + tp > 0):
        recall[s] = float(tp) / (fn + tp)

    print 'Evaluation computed for score threshold ' + str(score_thresholds[s])

#Plot object proposals
plt.plot(recall, precision)
plt.axis([0, 1, 0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Image containing text classification')
plt.show()

print 'Done'
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
score_thresholds = [-1]
precision = np.zeros((score_thresholds.__len__()))
recall = np.zeros((score_thresholds.__len__()))

for s in range(0,score_thresholds.__len__()):
    positive_images = []
    negative_images = []

    for file in glob.glob("../../data/coco-text/images_evaluation/*.txt"):
        data = numpy.loadtxt(file,delimiter=" ")
        for r in range (0,len(data)):
            #If image score is above threshold, is positive
            if(data[r,1] > score_thresholds[s]):
                positive_images.append(int(data[r,0]))
            else:
                negative_images.append(int(data[r, 0]))

    # Retrieve images with text (legible + illegible)
    imgIds1 = ct.getImgIds(imgIds=ct.train,
                          catIds=[('legibility', 'legible')])
    imgIds2 = ct.getImgIds(imgIds=ct.train,
                          catIds=[('legibility', 'illegible')])
    imgIds = imgIds1 + imgIds2

    tp = 0; fp = 0; fn = 0;

    for i in range(0, len(positive_images)):
        if positive_images[i] in imgIds:
            tp += 1
        else:
            fp += 1

    for i in range(0,len(negative_images)):
        if negative_images[i] in imgIds:
            fn += 1


    if(tp + fn > 0):
        precision[s] = tp / (tp + fp)
    if (fn + tp > 0):
        recall[s] = tp / (fn + tp)


#Plot object proposals
plt.plot(recall, precision)
plt.axis([0, 1, 0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
print 'Done'
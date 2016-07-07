import glob, os
import numpy
import json
import sys
import matplotlib.pyplot as plt

sys.path.append('coco-text-api')
import coco_evaluation
import coco_text
import numpy as np

badAnnotatedIds = [3992,4823,7124,8055,8439,10981,21097,21213,32626,8644]

ct = coco_text.COCO_Text('../../data/coco/coco-text-annotations/COCO_Text.json')

# Retrieve images with legible
imgIds = ct.getImgIds(imgIds=ct.val,
                       catIds=[('legibility', 'legible')])

score_thresholds = np.arange(0.96002,1.0002,0.0002)
# score_thresholds = np.arange(0,1.001,0.005)
precision = np.ones((score_thresholds.__len__()))
recall = np.zeros((score_thresholds.__len__()))

for s in range(0,score_thresholds.__len__()):

    print 'Evaluating for score threshold ' + str(score_thresholds[s]) + ' ...'

    positive_images = []
    negative_images = []

    #Classify images with scores
    cont = 0
    for file in glob.glob("../../data/coco-text/images_evaluation_/*.txt"):

        cont += 1
        # Only evaluate 500 images
        # if cont == 200:
        #      break

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
        if (positive_images[i] in badAnnotatedIds):
            continue
        elif (positive_images[i] in imgIds):
            tp += 1
        else:
            fp += 1
            #print positive_images[i]

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
    #print fp
    print 'Evaluation computed'


recall = recall[::-1]
precision = precision[::-1]

# Save evaluation data
filename = 'plot_data/coco-PR-textProposals.txt'
np.savetxt(filename, np.c_[recall,precision])

#Plot
plt.plot(recall, precision)
plt.axis([0, 1, 0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Image containing text classification')
plt.show()

print 'Done'
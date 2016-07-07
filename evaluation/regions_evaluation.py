import glob, os
import numpy
import json
import sys
import matplotlib.pyplot as plt

sys.path.append('coco-text-api')
import coco_evaluation
import coco_text
import numpy as np


print 'text'

score_thresholds = np.arange(0,1.01,0.01) #For textProposals should be more exhaustive than 0.01.. but they won't have same size!

#Edited thresholds for text proposals
# score_thresholds = [ 0.0,  0.02,   0.04,  0.06,  0.08,
#           0.1 ,   0.12,   0.14,  0.15,  0.16,  0.17,
#         0.18,  0.19,  0.2 ,  0.21,  0.22,  0.23,  0.24,  0.25,  0.26,
#         0.27,  0.28,  0.29,  0.3 ,  0.31,  0.32,  0.33,  0.34,  0.35,
#         0.36,  0.37,  0.38,  0.39,  0.4 ,  0.41,  0.42,  0.43,  0.44,
#         0.45,  0.46,  0.47,  0.48,  0.49,  0.5 ,  0.51,  0.52,  0.53,
#         0.54,  0.55,  0.56,  0.57,  0.58,  0.59,  0.6 ,  0.61,  0.62,
#         0.63,  0.64,  0.65,  0.66,  0.67,  0.68,  0.69,  0.7 ,  0.71,
#         0.72,  0.73,  0.74,  0.75,  0.76,  0.77,  0.78,  0.79,  0.8 ,
#         0.81,  0.82,  0.83,  0.84,  0.85,  0.86,  0.87,  0.88,  0.89,
#         0.9 ,  0.91,  0.92,  0.93,  0.94,  0.95,  0.96,  0.97, 0.975,  0.98,
#         0.985, 0.99, 0.993,  0.995, 0.996, 0.997,  0.998, 1.  ]

tp = np.zeros((score_thresholds.__len__()))
fn = np.zeros((score_thresholds.__len__()))

objects_detected = np.zeros((score_thresholds.__len__()))
objects_proposed = np.zeros((score_thresholds.__len__()))

print 'Loading annotations...'
ct = coco_text.COCO_Text('../../data/coco/coco-text-annotations/COCO_Text.json')

# Read all regions data
cont = 0

for file in glob.glob("../../data/coco-text/regions_evaluation/*.txt"):

            # Clean regions data
            regions_data = []

            #Print index
            cont +=  1
            if cont % 100 is 0:
                print 'Evaluating ' +  str(cont)

            #Only evaluate X images
            #if cont == 500:
                #break

            data = numpy.loadtxt(file,delimiter=" ")
            for r in range (0,len(data)):

                    region_data = {
                        'image_id': int(file[file.__len__() -16:file.__len__() -4]),
                        'category_id': 91,
                        'bbox': [int(data[r,0]), int(data[r,1]), int(data[r,2]), int(data[r,3]), 0],
                        'score': data[r,6],
                        'utf8_string': 'text',
                        'readable': -1,
                        'printed': -1,
                        'language': 'en',
                        'tp': 0,
                        'my_id': r
                    }

                    regions_data.append(region_data)

            # Load data needed to evaluate all the regions of the image
            our_results = ct.loadRes(regions_data)
            our_detections = coco_evaluation.getDetections(ct, our_results, detection_threshold=0.3)

            for r in range (0,our_detections['true_positives'].__len__()):
                id = our_detections['true_positives'][r]['my_id']
                regions_data[id]['tp'] = 1
                #print id
                #print regions_data[regions_data.__len__()-1-id]['score']

            image_id = regions_data[0]['image_id']

            for s in range(0, score_thresholds.__len__()):

                #Discard regions below threshold
                regions_data = list((d for d in regions_data if d['score'] > score_thresholds[s]))
                objects_proposed_image = regions_data.__len__()

                #Count tp from the remaining regions
                tp_image = list((d for d in regions_data if d['tp'] > 0)).__len__()

                #Compute fn as number annotated regions - tp
                fn_image = ct.imgToAnns[image_id].__len__() - tp_image

                # Accumulate evaluation data
                tp[s] = tp[s] + tp_image
                fn[s] = fn[s] + fn_image
                objects_proposed[s] = objects_proposed[s] + objects_proposed_image



#Compute ploting data
for s in range(0, score_thresholds.__len__()):
    if((tp[s] + fn[s]) > 0):
        objects_detected[s] = float(tp[s]) / (tp[s] + fn[s])
    objects_proposed[s] = objects_proposed[s] / cont #Total num regions / images evaluated


#Plot object proposals
print 'Ploting results'
objects_proposed = objects_proposed[::-1]
objects_detected = objects_detected[::-1]

# Save evaluation data
filename = 'plot_data/coco-OP-mean-0.3.txt'
np.savetxt(filename,np.c_[objects_proposed,objects_detected])

plt.plot(objects_proposed, objects_detected)
plt.ylim(0, 1)
plt.xscale('log')
plt.xlabel('# of proposals')
plt.ylabel('Detection rate')
plt.show()

print 'Done'



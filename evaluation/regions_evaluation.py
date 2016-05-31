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
score_thresholds = [-1, 0, 1, 2]
objects_detected = np.zeros((score_thresholds.__len__()))
objects_proposed = np.zeros((score_thresholds.__len__()))

for s in range(0,score_thresholds.__len__()):
        regions_data = []
        for file in glob.glob("../../data/coco-text/regions_evaluation/*.txt"):
            data = numpy.loadtxt(file,delimiter=" ")
            for r in range (0,len(data)):

                #If region score is above threshold, consider it
                if(data[r,4] > score_thresholds[s]):
                    region_data = {
                        'image_id': int(file[file.__len__() -16:file.__len__() -4]),
                        'category_id': 91,
                        'bbox': [int(data[r,0]), int(data[r,1]), int(data[r,2]), int(data[r,3]), 0],
                        'score': data[r,4],
                        'utf8_string': 'text',
                        'readable': -1,
                        'printed': -1,
                        'language': 'en'
                    }
                    regions_data.append(region_data)

        filename = 'regions_json/regions_data_' + str(score_thresholds[s]) + '.json'
        out_file = open(filename,"w")
        json.dump(regions_data, out_file)
        out_file.close()
        print 'Regions data computed for score threshold ' + str(score_thresholds[s])

        if len(regions_data) > 0:
            our_results = ct.loadRes(filename)

            our_detections = coco_evaluation.getDetections(ct, our_results, detection_threshold = 0.5)
            if(our_detections['true_positives'].__len__() + our_detections['false_negatives'].__len__() > 0):
                objects_detected[s] = float(our_detections['true_positives'].__len__()) / (our_detections['true_positives'].__len__() + our_detections['false_negatives'].__len__())
            # objects_proposed[s] = our_detections['true_positives'].__len__() + our_detections['false_positives'].__len__()
            objects_proposed[s] = len(regions_data)
#Plot object proposals
plt.plot(objects_proposed, objects_detected)
plt.axis([0, objects_detected.max(), 0, 1])
plt.xlabel('# of proposals')
plt.ylabel('Detection rate')
plt.show()
print 'Done'



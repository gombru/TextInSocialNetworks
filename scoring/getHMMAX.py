import numpy as np
from collections import defaultdict
import cv2
import glob
import time


with open('../../data/coco-text/images_evaluation/hm_max.txt', 'w') as f:
    start = time.time()
    count = 0
    for file in glob.glob("../../data/coco-text/regions_evaluation_coco_paper/*.txt"):
        count += 1
        print count
        hmfname = '../../data/coco-text/heatmaps-withoutIllegible/' + file.split('/')[5][:-4] + '.png'
        heatMap = cv2.imread(hmfname,cv2.IMREAD_GRAYSCALE)/255.0
        #Get heatmat maximum
        maxProb = heatMap.max()
        line = file.split('/')[5][:-4][15:] + ' ' + str(maxProb) + '\n'
        f.write(line)
    f.close()

end = time.time()
print 'Total time elapsed in heatmap computations'
print(end - start)
print 'Time per image'
print(end - start)/count
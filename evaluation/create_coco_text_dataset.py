import sys
sys.path.append('coco-text-api')
import coco_evaluation
import coco_text
import numpy as np
import skimage.io as io
import shutil

ct = coco_text.COCO_Text('../../data/coco/coco-text-annotations/COCO_Text.json')
imgIds = ct.getImgIds(imgIds=ct.train)
dataDir = '/home/imatge/caffe-master/data/coco/train2014/'
destDataDir = '/home/imatge/caffe-master/data/coco-text/train2014/'

for x in range(0,len(imgIds)):
    img = ct.loadImgs(imgIds[x])[0]
    print dataDir + img['file_name']
    shutil.copy2(dataDir + img['file_name'], destDataDir + img['file_name'])
print 'Done'

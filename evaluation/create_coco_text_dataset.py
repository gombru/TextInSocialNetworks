import sys
sys.path.append('coco-text-api')
import coco_evaluation
import coco_text
import numpy as np
import skimage.io as io
import shutil

ct = coco_text.COCO_Text('../../data/coco/coco-text-annotations/COCO_Text.json')

#All images
#imgIds = ct.getImgIds(imgIds=ct.val)

#Only legible text images
#imgIds = ct.getImgIds(imgIds=ct.val, catIds=[('legibility', 'illegible')])

#Legible + illegibleT
# Retrieve images with text (legible + illegible)
# imgIds1 = ct.getImgIds(imgIds=ct.train,
#                        catIds=[('legibility', 'legible')])
# imgIds2 = ct.getImgIds(imgIds=ct.train,
#                        catIds=[('legibility', 'illegible')])
# imgIds = imgIds1 + imgIds2

#All but non-legible text images
imgIds_all = ct.getImgIds(imgIds=ct.val)
imgIds_illegible = ct.getImgIds(imgIds=ct.val,
                       catIds=[('legibility', 'illegible')])
imgIds = list(set(imgIds_all) - set(imgIds_illegible))


dataDir = '/home/imatge/caffe-master/data/coco-text/val2014/'
destDataDir = '/home/imatge/caffe-master/data/coco-text/val2014-withoutIllegible/'

for x in range(0,len(imgIds)):
    img = ct.loadImgs(imgIds[x])[0]
    # print dataDir + img['file_name']
    shutil.copy2(dataDir + img['file_name'], destDataDir + img['file_name'])
print 'Done'

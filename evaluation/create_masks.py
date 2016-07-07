import sys
sys.path.append('coco-text-api')

import coco_text
import numpy as np
import skimage.io as io
import shutil
import cv2


dataDir = '/home/imatge/caffe-master/data/coco-text/train2014/'
destDataDir = '/home/imatge/caffe-master/data/coco-text/all-masks-legible/'

ct = coco_text.COCO_Text('../../data/coco/coco-text-annotations/COCO_Text.json')

#Only legible text!
imgIds1 = ct.getImgIds(imgIds=ct.train, catIds=[('legibility', 'legible')])
imgIds2 = ct.getImgIds(imgIds=ct.val, catIds=[('legibility', 'legible')])
imgIds = imgIds1 + imgIds2


for cocoid in imgIds:
    im = ct.loadImgs(cocoid)[0]
    mask = np.zeros([im['height'], im['width'],1])

    gt_bboxes = ct.imgToAnns[cocoid]
    for gt_box_id in gt_bboxes:
        gt_box = ct.anns[gt_box_id]['bbox']
        mask[int(gt_box[1]):int(gt_box[1])+int(gt_box[3]), int(gt_box[0]):int(gt_box[0])+int(gt_box[2])] = 255
    cv2.imwrite(destDataDir + im['file_name'][:im['file_name'].__len__()-3] + 'png', mask)
print 'Done'

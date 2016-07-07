import sys
sys.path.append('coco-text-api')

import coco_text
import numpy as np
import skimage.io as io
import shutil
import cv2
import csv


dataDir = '/home/imatge/caffe-master/data/coco-text/val-whithoutIllegible/'
destDataDir = '/home/imatge/caffe-master/data/coco-text/gt/'

ct = coco_text.COCO_Text('../../data/coco/coco-text-annotations/COCO_Text.json')

#All but non-legible text images
# imgIds_all = ct.getImgIds(imgIds=ct.val)
# imgIds_illegible = ct.getImgIds(imgIds=ct.val,
#                        catIds=[('legibility', 'illegible')])
# imgIds = list(set(imgIds_all) - set(imgIds_illegible))
cont = 0

imgIds = ct.getImgIds(imgIds=ct.val,
                       catIds=[('legibility', 'legible')])

for cocoid in imgIds:

    # Print index
    cont += 1
    if cont % 1 is 0:
        print 'Doing ' + str(cont)

    im = ct.loadImgs(cocoid)[0]
    with open(destDataDir + im['file_name'][:im['file_name'].__len__()-3] + 'csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        gt_bboxes = ct.imgToAnns[cocoid]
        for gt_box_id in gt_bboxes:

            if (str(ct.anns[gt_box_id]['language']) == 'not english') and str(ct.anns[gt_box_id]['legibility']) == 'legible':
                ct.anns[gt_box_id]['utf8_string'] = '###'
            if str(ct.anns[gt_box_id]['legibility']) == 'illegible':
                ct.anns[gt_box_id]['utf8_string'] = '###'
                print 'illegible text'
            print ct.anns[gt_box_id]['utf8_string']

            if ct.anns[gt_box_id]['utf8_string'] == u'\xa9Northline':
                ct.anns[gt_box_id]['utf8_string'] = 'Northline'
            if ct.anns[gt_box_id]['utf8_string'] == u'\xa9NORTHLINE':
                ct.anns[gt_box_id]['utf8_string'] = 'NORTHLINE'
            if ct.anns[gt_box_id]['utf8_string'] == u'\xa9raarup.eu':
                ct.anns[gt_box_id]['utf8_string'] = 'raarup.eu'
            if ct.anns[gt_box_id]['utf8_string'] == u'59\xa2':
                ct.anns[gt_box_id]['utf8_string'] = '59'
            if ct.anns[gt_box_id]['utf8_string'] == u'79\xa2lb':
                ct.anns[gt_box_id]['utf8_string'] = '79lb'
            if ct.anns[gt_box_id]['utf8_string'] == u'HIST\xd3RICO':
                ct.anns[gt_box_id]['utf8_string'] = 'HISTORICO'
            if ct.anns[gt_box_id]['utf8_string'] == u'22\xb0n':
                ct.anns[gt_box_id]['utf8_string'] = '22n'
            if ct.anns[gt_box_id]['utf8_string'] == u'120\xb0E.':
                ct.anns[gt_box_id]['utf8_string'] = '120E.'
            if ct.anns[gt_box_id]['utf8_string'] == u'\xa31.50':
                ct.anns[gt_box_id]['utf8_string'] = '1.50'
            if ct.anns[gt_box_id]['utf8_string'] == u'L\xe4tta':
                ct.anns[gt_box_id]['utf8_string'] = 'Latta'
            if ct.anns[gt_box_id]['utf8_string'] == u'\u20ac1.40':
                ct.anns[gt_box_id]['utf8_string'] = '1.40'
            if ct.anns[gt_box_id]['utf8_string'] == u'Throat\u2022Medical':
                ct.anns[gt_box_id]['utf8_string'] = 'Throat Medical'
            if ct.anns[gt_box_id]['utf8_string'] == u'Care\u2022Urology':
                ct.anns[gt_box_id]['utf8_string'] = 'Care Urology'
            if ct.anns[gt_box_id]['utf8_string'] == u'79\xa2':
                ct.anns[gt_box_id]['utf8_string'] = '79'
            if ct.anns[gt_box_id]['utf8_string'] == u'\u20a4129':
                ct.anns[gt_box_id]['utf8_string'] = '129'
            if ct.anns[gt_box_id]['utf8_string'] == u'\u20ac3':
                ct.anns[gt_box_id]['utf8_string'] = '3'
            if ct.anns[gt_box_id]['utf8_string'] == u'\u20ac2':
                ct.anns[gt_box_id]['utf8_string'] = '2'
            if ct.anns[gt_box_id]['utf8_string'] == u'\u20ac329.00':
                ct.anns[gt_box_id]['utf8_string'] = '329.00'
            if ct.anns[gt_box_id]['utf8_string'] == u'\u20ac339.00':
                ct.anns[gt_box_id]['utf8_string'] = '339.00'
            if ct.anns[gt_box_id]['utf8_string'] == u'\u20ac199':
                ct.anns[gt_box_id]['utf8_string'] = '199'

            if ct.anns[gt_box_id]['utf8_string'] == u' \xa9':
                continue
            if ct.anns[gt_box_id]['utf8_string'] == u' \xd3':
                continue
            if ct.anns[gt_box_id]['utf8_string'] == u'\xe4':
                continue
            if ct.anns[gt_box_id]['utf8_string'] == u'\xa3' or ct.anns[gt_box_id]['utf8_string'] == u'\xa9':
                continue

            gt_box = ct.anns[gt_box_id]['bbox']
            text = str(ct.anns[gt_box_id]['utf8_string'])

            text = text.replace(',','.')

            spamwriter.writerow([int(gt_box[0]), int(gt_box[1]), int(gt_box[2]), int(gt_box[3]), text] )

print 'Done'

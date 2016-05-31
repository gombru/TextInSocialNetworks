import sys
sys.path.append('/home/imatge/caffe-master/python')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import caffe

#Compute heatmaps from images in txt
val = np.loadtxt('../../data/coco-text/miniTrain.txt', dtype=str)

# load net
net = caffe.Net('voc-fcn16s/deploy.prototxt', '../../data/fcn_training/snapshot/train_iter_364000.caffemodel', caffe.TEST)


for idx in val:

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    # im = Image.open('data/pascal/VOC2011/JPEGImages/2007_000129.jpg')
    im = Image.open('../../data/coco-text/miniTrain/' + idx + '.jpg')
    # im.thumbnail([640, 360], Image.ANTIALIAS)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    heatmap = net.blobs['score_conv'].data[0][1,:,:]
    heatmap = heatmap - heatmap.min()
    heatmap = (255.0 / heatmap.max() * (heatmap - heatmap.min())).astype(np.uint8)
    heatmap = Image.fromarray(heatmap)
    heatmap.save('/home/imatge/caffe-master/data/coco-text/heatmaps/' + idx + '.png')
    print 'Heatmap saved for image: ' +idx
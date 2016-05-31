import sys
sys.path.append('/home/imatge/caffe-master/python')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
# im = Image.open('data/pascal/VOC2011/JPEGImages/2007_000129.jpg')
im = Image.open('data/img_56.jpg')
im.thumbnail([640, 360], Image.ANTIALIAS)
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('voc-fcn16s/deploy.prototxt', 'voc-fcn16s/snapshot/train_iter_164000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score_conv'].data[0].argmax(axis=0)
plt.imshow(out)
print 'Done'
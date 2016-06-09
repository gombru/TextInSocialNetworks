import sys
sys.path.append('/home/imatge/caffe-master/python')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import caffe
import surgery, score
from scipy.misc import imresize, imsave, toimage



caffe.set_device(0)
caffe.set_mode_gpu()

#Compute heatmaps from images in txt
#val = np.loadtxt('../../data/icdar-resized/val.txt', dtype=str)
val = np.loadtxt('../../data/coco-text/val-withoutIllegible.txt', dtype=str)

globalMax = 13.12
globalMin = 8.2

# load net
net = caffe.Net('voc-fcn8s-atonce/deploy.prototxt', '../../data/fcn_training/coco-104000/train_iter_104000.caffemodel', caffe.TEST)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
for idx in val:

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    # im = Image.open('../../data/icdar-resized/ch4_training_images/' + idx + '.jpg')  #.resize((512, 512), Image.ANTIALIAS)
    im = Image.open('../../data/coco-text/val2014-withoutIllegible/' + idx + '.jpg')

    if (im.size.__len__() == 2):
        im_gray = im
        im = Image.new("RGB", im_gray.size)
        im.paste(im_gray)

    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # run net and take argmax for prediction
    net.forward()

    hmap_0 = net.blobs['score_conv'].data[0][0, :, :]
    hmap_1 = net.blobs['score_conv'].data[0][1, :, :]
    # hmap_0 = np.exp(hmap_0)
    # hmap_1 = np.exp(hmap_1)
    # hmap_softmax = hmap_1 / (hmap_0 + hmap_1)
    #
    # # Save color softmax heatmap
    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(5.12,5.12)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(hmap_softmax, aspect='auto', cmap="jet")
    # fig.savefig('/home/imatge/caffe-master/data/coco-text/heatmaps/' + idx + '-ht.jpg')
    # plt.close(fig)

    # Save grayscale heatmap
    # toimage(hmap_softmax, cmin=0.0, cmax=1.0).save('/home/imatge/caffe-master/data/icdar-resized/heatmaps/' + idx + '.png')

    # Save global normalized heatmap
    heatmap = net.blobs['score_conv'].data[0][1, :, :]
    heatmap = heatmap + globalMin

    # Check if the image values exceed 255. If that happend black pixels appear in text areas. Can update max but righ now, stop
    if heatmap.max() > globalMax:
        print 'Waring: Image ' + idx + 'exeeded 255 value'
        break

    heatmap = ((255.0 /globalMax) * heatmap).astype(np.uint8)
    heatmap = Image.fromarray(heatmap)
    heatmap.save('/home/imatge/caffe-master/data/coco-text/heatmaps/' + idx + '.png')

    print 'Heatmap saved for image: ' +idx


print 'Done'
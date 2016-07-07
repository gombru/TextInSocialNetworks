from PIL import Image
import glob, os



dataDir = '/home/imatge/caffe-master/data/icdar/ch4_BoundingBox_images/'
destDIR = '/home/imatge/caffe-master/data/icdar-resized/ch4_BoundingBox_images/'


for file in glob.glob(dataDir + '*.png'):
    im = Image.open(file).resize((512, 512), Image.ANTIALIAS)
    file = file.split('/')
    filename = file[file.__len__() - 1]
    im.save(destDIR + filename)

import glob, os

dataDir = '/home/imatge/caffe-master/data/coco-text/val-withoutIllegible/'

text_file = open("/home/imatge/caffe-master/data/coco-text/val-withoutIllegible.txt", "w")


for file in glob.glob(dataDir + '*.jpg'):
    file = file.split('/');
    filename = file[file.__len__() - 1]
    text_file.write(filename[:filename.__len__() - 4] + '\n')

text_file.close()
print 'Done'
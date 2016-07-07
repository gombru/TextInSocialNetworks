import glob, os
from random import randint

numIm = 50000

dataDir = '/home/imatge/caffe-master/data/synthtext/masks/'

text_file = open("/home/imatge/caffe-master/data/synthtext/train.txt", "w")

for idx in range(0,numIm):
    folder = randint(1,200)
    list = glob.glob(dataDir + str(folder) + '/*.png')
    index = randint(1 ,len(list) - 1)
    file = list[index].split('/')
    filename = file[file.__len__() - 1]
    text_file.write(str(folder) + '/' + filename[:filename.__len__() - 4] + '\n')

text_file.close()
print 'Done'
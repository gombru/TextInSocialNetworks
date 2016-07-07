import numpy as np
from collections import defaultdict
import cv2
import glob


def getConfidenceForAll(hm,prop):
    #output=np.empty([prop.shape[0],5])
    #print prop.shape
    #output[:,:4]=prop[:,:4]
    ihm=np.zeros([hm.shape[0]+1,hm.shape[1]+1])
    ihm[1:,1:]=hm.cumsum(axis=0).cumsum(axis=1) #integral image
    confidenceDict=defaultdict(list)
    for rectId in range(prop.shape[0]):
        rect=tuple(prop[rectId,:4])
        (l,t,w,h)=rect
        r=l+w#the coordinates are translated by 1 because of ihm zero pad
        b=t+h
        #output[rectId,4]=((ihm[b,r]+ihm[t,l])-(ihm[b,l]+ihm[t,r]))/(w*h)
        confidenceDict[rect].append(((ihm[b,r]+ihm[t,l])-(ihm[b,l]+ihm[t,r]))/(w*h))
    res=np.array([tup[1]+(tup[0],) for tup in sorted([(max(confidenceDict[rec]),rec) for rec in confidenceDict.keys()],reverse=True)])
    return res

with open('../../data/coco-text/images_evaluation/hm_tp_fusion_05.txt', 'w') as f:

    for file in glob.glob("../../data/coco-text/regions_evaluation_coco_paper/*.txt"):

        hmfname = '../../data/coco-text/heatmaps-withoutIllegible/' + file.split('/')[5][:-4] + '.png'
        heatMap = cv2.imread(hmfname,cv2.IMREAD_GRAYSCALE)/255.0
        proposals= np.genfromtxt(file, delimiter=' ')
        res = getConfidenceForAll(heatMap,proposals)
        #Get maximum probability
        res = 0.5 * res[:,4] + 0.5 * proposals[:,4]
        maxProb = max(res)
        line = file.split('/')[5][:-4] + ' ' + str(maxProb) + '\n'
        f.write(line)
    f.close()



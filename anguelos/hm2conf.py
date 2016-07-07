#!/usr/bin/env python
import sklearn.naive_bayes


from multiprocessing import Pool
import os.path

import numpy as np
from commands import getoutput as go
import sys
import time
import numpy.matlib
from collections import defaultdict

sys.path.append('/home/anguelos/work/projects/opencv_gsoc/build/lib/')
import cv2
import matplotlib.pyplot as plt

def get2PointIU(gtMat,resMat):
    maxProposalsIoU=int(switches['maxProposalsIoU'])
    if maxProposalsIoU>0:
        resMat=resMat[:maxProposalsIoU,:]
    #matSz=(gtMat.shape[0],resMat.shape[0])
    gtLeft=numpy.matlib.repmat(gtMat[:,0],resMat.shape[0],1)
    gtTop=numpy.matlib.repmat(gtMat[:,1],resMat.shape[0],1)
    gtRight=numpy.matlib.repmat(gtMat[:,0]+gtMat[:,2]-1,resMat.shape[0],1)
    gtBottom=numpy.matlib.repmat(gtMat[:,1]+gtMat[:,3]-1,resMat.shape[0],1)
    gtWidth=numpy.matlib.repmat(gtMat[:,2],resMat.shape[0],1)
    gtHeight=numpy.matlib.repmat(gtMat[:,3],resMat.shape[0],1)

    resLeft=numpy.matlib.repmat(resMat[:,0],gtMat.shape[0],1).T
    resTop=numpy.matlib.repmat(resMat[:,1],gtMat.shape[0],1).T
    resRight=numpy.matlib.repmat(resMat[:,0]+resMat[:,2]-1,gtMat.shape[0],1).T
    resBottom=numpy.matlib.repmat(resMat[:,1]+resMat[:,3]-1,gtMat.shape[0],1).T
    resWidth=numpy.matlib.repmat(resMat[:,2],gtMat.shape[0],1).T
    resHeight=numpy.matlib.repmat(resMat[:,3],gtMat.shape[0],1).T

    intL=np.max([resLeft,gtLeft],axis=0)
    intT=np.max([resTop,gtTop],axis=0)

    intR=np.min([resRight,gtRight],axis=0)
    intB=np.min([resBottom,gtBottom],axis=0)

    intW=(intR-intL)+1
    intW[intW<0]=0

    intH=(intB-intT)+1
    intH[intH<0]=0
    
    I=intH*intW
    U=resWidth*resHeight+gtWidth*gtHeight-I
    IoU=I/(U+.0000000001)
    return (IoU,I,U)

#filename conversions
def createRequiredDirs(filenameList,fromFilesDir):
    filesDirList=set(['/'.join(f.split('/')[:-1]) for f in filenameList])
    if fromFilesDir[0]=='+':
        for fd in filesDirList:
            addP=fromFilesDir[1:].split('/')
            addP[-1]=addP[-1]+fd.split('/')[-1]
            go('mkdir -p '+fd+'/'+'/'.join(addP))
    else:
        for fd in filesDirList:
            go('mkdir -p '+fd+'/'+fromFilesDir)


def getInputFromConf(confFname):
    pathList=confFname.split('/')
    pathList[-2]='input'
    return '/'.join(pathList)[:-3]+'jpg'


def getThresholdFromHm(hmFname,outDir):
    pathList=hmFname.split('/')
    pathList[-2]=outDir+pathList[-2]
    return '/'.join(pathList)[:-3]+'csv'


def getProposalFromConf(hmFname):
    pathList=hmFname.split('/')
    pathList[-2]='proposals'
    return '/'.join(pathList)[:-3]+'csv'


def getConfFromHm(hmFname):
    pathList=hmFname.split('/')
    pathList[-2]='proposals'
    return '/'.join(pathList)[:-3]+'csv'


def getIouFromConf(confFname):
    pathList=confFname.split('/')
    pathList[-2]='iou_'+pathList[-2]
    return '/'.join(pathList)[:-3]+'png'


def getGtFromConf(imageFname):
    pathList=imageFname.split('/')
    pathList[-2]='gt'
    return '/'.join(pathList)[:-3]+'txt'


def getProposalFromImage(imageFname):
    pathList=imageFname.split('/')
    pathList[-2]='proposals'
    return '/'.join(pathList)[:-3]+'csv'

def getProposalFromHeatmap(heatmapFname):
    if heatmapFname[-3:]=='png':
        heatmapFname=heatmapFname[:-3]+'csv'
    pathList=heatmapFname.split('/')
    pathList[-2]='proposals'
    return '/'.join(pathList)

def getConfidenseFromHeatmap(heatmapFname):
    if heatmapFname[-3:]=='png':
        heatmapFname=heatmapFname[:-3]+'csv'
    pathList=heatmapFname.split('/')
    pathList[-2]='conf_'+pathList[-2]
    return '/'.join(pathList)

getConfidenseFromProposal=getConfidenseFromHeatmap

#str 2 numpy
def arrayToCsvStr(arr):
    if len(arr.shape)!=2:
        raise Exception("only 2D arrays")
    resLines=[list(row) for row in list(arr)]
    return '\n'.join([','.join([str(col) for col in row])  for row in resLines])

def csvStr2Array(csvStr):
    return np.array([[float(c) for c in l.split(',')] for l in csvStr.split('\n') if len(l)>0])

def fname2Array(fname):
    if fname[-3:]=='png':
        print "FROM PNG:",fname
        return cv2.imread(fname,cv2.IMREAD_GRAYSCALE)/255.0
    else: #assuming csv
        print "FROM CSV"
        res= np.genfromtxt(fname, delimiter=',')
        print res[:8,:]
        return res


def array2csvFname(arr,csvFname):
    np.savetxt(csvFname,arr, '%9.5f',delimiter=',')

def array2pngFname(arr,pngFname):
    cv2.imwrite(pngFname,(arr*255).astype('uint8'),[cv2.IMWRITE_PNG_COMPRESSION ,0])


def loadTxtGtFile(fname):
    lines=[l.strip().split(',') for l in open(fname).read().split('\n') if len(l)]
    if len(lines)==0:
        return (np.array([5,5,3,3],dtype='float'),['###'])
    if lines[0][0][:3]=='\xef\xbb\xbf':
        lines[0][0]=lines[0][0][3:]
    if len(lines[0])>8:#4 points
        rects=np.empty([len(lines),4],dtype='float')
        tmpArr=np.array([[int(c) for c in line[:8]] for line in lines])
        left=tmpArr[:,[0,2,4,6]].min(axis=1)
        right=tmpArr[:,[0,2,4,6]].max(axis=1)
        top=tmpArr[:,[1,3,5,7]].min(axis=1)
        bottom=tmpArr[:,[1,3,5,7]].max(axis=1)
        rects[:,0]=left
        rects[:,1]=top
        rects[:,2]=1+right-left
        rects[:,3]=1+bottom-top
        print 'RECTS:',rects.shape
        trans=[','.join(line[8:]) for line in lines]
    else:#ltwh
        rects=np.array([[int(c) for c in line[:4]] for line in lines],dtype='float')
        trans=[','.join(line[4:]) for line in lines]
    return (rects,trans)


def getDontCare(transcriptions,dictionary=[]):
    dictionary=set(dictionary)
    if len(dictionary)==0:
        return np.array([tr!='###' for tr in transcriptions],dtype='bool')
    else:
        return np.array([(tr in dictionary) for tr in transcriptions],dtype='bool')


#algorithm
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

switches={'maxProposalsIoU':'20000',#IoU over this are not computed
'threads':'1',
'dontCareDictFile':'',
'IoUThresholds':'[.5]',
'extraPlotDirs':'{".":"Confidence"}',
'care':'True', #If true dont cares are supressed
'bayesianFname':'/tmp/bayesian/',
'plotter':'plt.semilogx',
'thr':'0.1'
}


if __name__=='__main__':
    hlp="""
    hm2conf blabla/*/heatma.../*.csv 
    hm2conf blabla/*/heatma.../*.png 
    img2prop blabla/*/input/*.jpg 
    """
    params=[(len(p)>0 and p[0]!='-',p) for p in sys.argv]
    sys.argv=[p[1] for p in params if p[0]]
    switches.update(dict([p[1][1:].split('=') for p in params if not p[0]]))
    print 'Threads',int(switches['threads'])
    
    if sys.argv[1]=='hm2conf':
        createRequiredDirs(sys.argv[2:],'+../conf_')
        def worker(heatmapFname):
            t=time.time()
            heatMap=fname2Array(heatmapFname)
            #proposals=csvStr2Array(open(getProposalFromHeatmap(heatmapFname)).read())
            proposals=fname2Array(getProposalFromHeatmap(heatmapFname))
            print 'heatmap:',heatMap.shape,'\tproposals:',proposals.shape
            confArray=getConfidenceForAll(heatMap,proposals)
            oFname=getConfidenseFromHeatmap(heatmapFname)
            #go('mkdir -p '+'/'.join(oFname.split('/')[:-1]))
            array2csvFname(confArray,oFname)
            print heatmapFname, ' to ', getConfidenseFromHeatmap(heatmapFname), ' ', int((time.time()-t)*1000)/1000.0,' sec.'
            return None
        pool=Pool(int(switches['threads']))
        print pool.map(worker,sys.argv[2:])
        #pool.join()
        sys.exit(0)

    if sys.argv[1]=='img2prop':
        proposalCmdPath='/home/anguelos/work/projects/opencv_gsoc/TextProposals-master/img2hierarchy'
        if not os.path.isfile(proposalCmdPath):
            print 'img2prop needs ',proposalCmdPath,' compiled!'
            sys.exit(1)
        def worker(imageFname):
            t=time.time()
            open(getProposalFromImage(imageFname),'w').write(go(proposalCmdPath+' '+imageFname))
            print imageFname, ' to ', getProposalFromImage(imageFname), ' ', int((time.time()-t)*1000)/1000.0,' sec.'
        createRequiredDirs(sys.argv[2:],'../proposals')
        pool=Pool(int(switches['threads']))
        print pool.map(worker,sys.argv[2:])
        sys.exit(0)

    if sys.argv[1]=='conf2IoU':
        createRequiredDirs(sys.argv[2:],'+../iou_')
        if switches['dontCareDictFile']!='':
            dictionary=[l for l in open(switches['dontCareDictFile']).read().split('\n') if len(l)]
        else:
            dictionary=[]
        def worker(confFname):
            t=time.time()
            confMat=fname2Array(confFname)
            idx=np.argsort(-confMat[:,4])
            confMat=confMat[idx,:]
            #confMat=confMat[reversed(idx),:]#UGLY!!!!!!!!!!!!!!!!!!!!!!!!!!
            gtMat,transcriptions=loadTxtGtFile(getGtFromConf(confFname))
            dontCare=getDontCare(transcriptions,dictionary)
            (IoU,I,U)=get2PointIU(gtMat,confMat)
            augmentedIoU=np.empty([IoU.shape[0]+1,IoU.shape[1]])
            augmentedIoU[:-1,:]=IoU
            augmentedIoU[-1,:]=dontCare
            #print 'Transcriptions: ',transcriptions
            #print 'DONT CARE: ',dontCare
            #array2csvFname(IoU,getIouFromConf(confFname))
            array2pngFname(augmentedIoU,getIouFromConf(confFname))
            #open(getIouFromConf(confFname),'w').write(arrayToCsvStr(IoU))
            print confFname, ' to ', getIouFromConf(confFname), ' ', int((time.time()-t)*1000)/1000.0,' sec.'
            return None
        pool=Pool(int(switches['threads']))
        print pool.map(worker,sys.argv[2:])
        sys.exit(0)


    if sys.argv[1]=='hmThr':
        outDir='../conf_thr%02d_'%int(eval(switches['thr'])*100)
        def worker(hmFname):
            thr=eval(switches['thr'])
            thrFname=getThresholdFromHm(hmFname,thr)
            proposals=fname2Array(getProposalFromConf(hmFname))[:,:5]
            hmconf=fname2Array(getConfFromHm(hmFname))
            #propDict=defaultdict(list)
#            hmDict=defaultdict(list)
            hmDict=dict([(tuple(hmconf[k,:4]),hmconf[k,4]) for k in range(hmconf.shape[0])])
            propDict=dict([(tuple(proposals[k,:4]),proposals[k,4]) for k in range(proposals.shape[0])])
            if set(hmDict.keys())!=set(propDict.keys()):
                raise Exception("")
            weakThrMat=np.empty([len(hmDict.keys()),6])
            rectList=hmDict.keys()
            for rectId in range(len(hmDict)):
                r=rectList[rectId]
                weakThrMat[rectId,:]=r+(propDict[r],hmDict[r])
            weakThrMat[:,4]*=(weakThrMat[:,5]>thr)
            idx=np.argsort(-weakThrMat[:,4])
            weakThrMat=weakThrMat[idx,:]
            array2csvFname(weakThrMat,thrFname)
        createRequiredDirs(sys.argv[2:],'+'+outDir)
        pool=Pool(int(switches['threads']))
        print pool.map(worker,sys.argv[2:])
        #for f in sys.argv[2:]:
        #    worker(f)
        sys.exit(0)



    if sys.argv[1]=='prop2conf':
        def worker(propFname):
            proposals=fname2Array(propFname)
            propRects=proposals[:,:4].astype('int32')
            confidenceDict=defaultdict(list)
            for rectId in range(proposals.shape[0]):
                rect=tuple(propRects[rectId,:])
                confidenceDict[rect].append(proposals[rectId,4])
            newProds=reversed(sorted([(max(confidenceDict[rec]),rec) for rec in confidenceDict.keys()]))
            arr=np.array([list(s[1])+list([s[0]]) for s in newProds],dtype='float')
            array2csvFname(arr,getConfidenseFromProposal(propFname))
        createRequiredDirs(sys.argv[2:],'+../conf_')
        pool=Pool(int(switches['threads']))
        print pool.map(worker,sys.argv[2:])
        #for fname in sys.argv[2:]:
        #    worker(fname)
        sys.exit(0)



    if sys.argv[1]=='icdar2normGt':
        for gtFname in sys.argv[2:]:
            gtMat,transcriptions=loadTxtGtFile(gtFname)
            gtOut=np.zeros([gtMat.shape[0],5])
            gtOut[:,:4]=gtMat
            res=[]
            for k in range(gtMat.shape[0]):
                res.append(','.join([str(l) for l in  gtMat[k,:4].astype('int32')])+','+transcriptions[k])
            #array2csvFname(gtOut,gtFname[:-3]+'csv')
        sys.exit(0)


    if sys.argv[1]=='getCumRecall':
        def getConfFromConf(fname,confName):
            if confName=='.':
                return fname
            else:
                return '/'.join(fname.split('/')[:-2]+[confName]+fname.split('/')[-1:])
        iouThr=eval(switches['IoUThresholds'])
        maxProposals=eval(switches['maxProposalsIoU'])
        #resGtObjDetected=np.zeros([maxProposals,len(iouThr)],dtype='int64')
        resGtObjCount=0
        confDict=eval(switches['extraPlotDirs'])
        resGtObjDetected=dict([(k,np.zeros([maxProposals,len(iouThr)],dtype='int64')) for k in confDict.keys()])
        for confFname in sys.argv[2:]:
            for confStr in confDict.keys():
                print getConfFromConf(confFname,confStr)
                augmentedIoU=fname2Array(getIouFromConf(getConfFromConf(confFname,confStr)))
                care=augmentedIoU[-1,:]
                keepProposals=min(maxProposals,augmentedIoU.shape[0]-1)
                if eval(switches['care']):
                    IoU=augmentedIoU[:keepProposals,care.astype('bool')]
                else:
                    IoU=augmentedIoU[:keepProposals,:]
                print care
                print IoU.shape
                resGtObjCount+=IoU.shape[1]
                #dontCare=augmentedIoU[:-1,:]
                #IoU[:,np.argmax(IoU,axis=1)]*=2#removin non maximal matches (a proposal identifies at most an object)
                #IoU-=augmentedIoU[:keepProposals,:]
                for tNum in range(len(iouThr)):
                    thr=iouThr[tNum]
                    found=((IoU>=thr).cumsum(axis=0)>0)
                    resGtObjDetected[confStr][:keepProposals,tNum]+=found.sum(axis=1)
                    try: #needed for when no match above the ratio occurs eg COCO_train2014_000000000036
                        resGtObjDetected[confStr][keepProposals:,tNum]+=found.sum(axis=1).max()
                    except ValueError:
                        pass
                print IoU[:3,:]
        sortedKeysByLegends=[e[1][0] for e in sorted([(confDict[k],(k,confDict[k])) for k in confDict.keys()])]
        for confStr in sortedKeysByLegends:
            #plt.plot(resGtObjDetected[confStr].astype('float')/(resGtObjCount/len(confDict.keys())),label=confDict[confStr])
            eval(switches['plotter'])(resGtObjDetected[confStr].astype('float')/(resGtObjCount/len(confDict.keys())),label=confDict[confStr])
            #plt.legend(confDict[confStr])
        plt.legend()
        plt.show()
    sys.exit(0)

    if sys.argv[1]=='dbgIoU':
        for confFname in sys.argv[2:]:
            IoU=fname2Array(getIouFromConf(confFname))
            dontCare=IoU[-1,:]
            gt,transcr=loadTxtGtFile(getGtFromConf(confFname))
            gt=gt.astype('int32')
            conf=fname2Array(confFname).astype('float')
            print conf[:10,:]
            image=cv2.imread(getInputFromConf(confFname),cv2.IMREAD_COLOR)
            gtImage=image.copy()
            for k in range(gt.shape[0]):
                if not dontCare[k]:
                    gtImage=cv2.rectangle(gtImage,(gt[k,0],gt[k,1]),(gt[k,0]+gt[k,2],gt[k,1]+gt[k,3]),(255,0,255))
                else:
                    gtImage=cv2.rectangle(gtImage,(gt[k,0],gt[k,1]),(gt[k,0]+gt[k,2],gt[k,1]+gt[k,3]),(0,0,255))
            cv2.namedWindow('DBG')
            print gt
            c=conf.astype('int32')
            for k in range(100):
                fgImage=gtImage.copy()
                print 'PROP # ',k,'  ',IoU[k,:].max(),' ',transcr[np.argmax(IoU[k,:])],' @ ',np.argmax(IoU[k,:])
                print conf[k,:]
                fgImage=cv2.rectangle(fgImage,(c[k,0],c[k,1]),(c[k,0]+c[k,2],c[k,1]+c[k,3]),(0,255,0))
                cv2.imshow('DBG',fgImage);cv2.waitKey()


    if sys.argv[1]=='trainBayesian':
        bayes=GaussianNB()
        matList=[]
        for gtFname in sys.argv[2:]:
            gtMat,transcriptions=loadTxtGtFile(gtFname)
            gtMat.append(gtMat)
        nbObjects=sum([gtMat.shape[0] for gtMat in matList])
        outputData

    print hlp
    print "unrecognised mode "+sys.argv[1]+" Aborting"
sys.exit(1)

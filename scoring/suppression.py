import numpy as np
import glob, os
from numpy import genfromtxt


coeff = 0.05 #Below regions are suppressed
count = 0
for file in glob.glob("../../data/icdar/proposals/*.csv"):

    count += 1
    print count
    fcn = genfromtxt('../../data/icdar/conf_voc-fcn8s-atonce-104000it/' + file.split('/')[5], delimiter=',')
    regions_data = {}
    output_data = {}

    for i in range (0,fcn.shape[0]):
        rect = (int(fcn[i,0]), int(fcn[i,1]), int(fcn[i,2]), int(fcn[i,3]))
        regions_data[rect] = fcn[i, 4] #FCN score
        output_data[rect] = 0 #FCN score

    weak = genfromtxt(file, delimiter=',')
    weak[:, 4] = weak[:,4] * -1

    for i in range(0, weak.shape[0]):
        rect = (int(weak[i,0]), int(weak[i,1]), int(weak[i,2]), int(weak[i,3]))
        if regions_data[rect] > coeff and weak[i,4] > output_data[rect]: #If is above energy TH and has not grater value keep weak class score
            output_data[rect] = weak[i,4]

    i = 0
    for key, value in output_data.iteritems():
        fcn[i,0] = int(key[0])
        fcn[i,1] = int(key[1])
        fcn[i,2] = int(key[2])
        fcn[i,3] = int(key[3])
        fcn[i,4] = value
        i += 1
        #print value


    fcn = fcn[fcn[:, 4].argsort()[::-1]] #Top ranked go first
    directory = '../../data/icdar/conf_voc-fcn8s-atonce-104000it_suppression_005/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savetxt(directory + file.split('/')[5], fcn, delimiter=",")

print 'Done'
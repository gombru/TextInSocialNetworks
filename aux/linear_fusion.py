import numpy as np
import glob, os



coef = 0.5 #Multiplied by weak classifier
count = 0
for file in glob.glob("../../data/icdar/proposals/*.csv"):
    count += 1

    index = file.split('/').__len__()
    fcn = np.genfromtxt('../../data/icdar/conf_voc-fcn8s-atonce-104000it/' + file.split('/')[5], delimiter=',')
    regions_data = {}
    aux_data = {}
    out_data = {}

    for i in range (0,fcn.shape[0]):
        rect = (int(fcn[i,0]), int(fcn[i,1]), int(fcn[i,2]), int(fcn[i,3]))
        regions_data[rect] = fcn[i, 4] #FCN score
        aux_data[rect] = 0
        out_data[rect] = 0

    weak = np.genfromtxt(file, delimiter=',')
    weak[:, 4] = weak[:,4] * -1

    for i in range(0, weak.shape[0]):
        rect = (int(weak[i,0]), int(weak[i,1]), int(weak[i,2]), int(weak[i,3]))
        if weak[i,4] > aux_data[rect]: #Keep best score for each region in case of duplicates
            aux_data[rect] = weak[i,4]
            out_data[rect] =  coef * weak[i,4] + (1 - coef) * regions_data.get(rect)

    i = 0
    out = np.zeros([fcn.shape[0], 5])
    for key, value in out_data.iteritems():
        out[i,0] = int(key[0])
        out[i,1] = int(key[1])
        out[i,2] = int(key[2])
        out[i,3] = int(key[3])
        out[i,4] = value
        i += 1

    out = out[out[:, 4].argsort()[::-1]] #Top ranked go first

    # Create dir and save data
    directory = '../../data/icdar/conf_voc-fcn8s-atonce-104000it_fusion_05/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savetxt(directory + file.split('/')[index - 1], out, delimiter=",")

    print file.split('/')[index - 1]

if count == 0:
    print 'No files found in that folder'
else:
    print 'Done for ' + str(count) + ' files.'

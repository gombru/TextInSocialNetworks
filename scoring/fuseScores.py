import numpy as np
import glob, os
import sys

# ARGS:
# Fuse method: linear / suppression
# Database
# Net name
# Fusing coffecient [0-1]

#Performs supresion. Args: Name of the net, coef of supresion (regions with lower mean are suppresed: 0-1)
def suppression(database, net, coef):
    #coeff = 0.05 # Regions with a lower mean probability in heatmap are suppressed
    print coef
    coef = float(coef)
    count = 0

    for file in glob.glob('../blabla' +database+'/proposals/*.csv'):
        count += 1

        index = file.split('/').__len__()
        fcn = np.genfromtxt('../blabla' +database+'/conf_' +net+ '/' + file.split('/')[index-1], delimiter=',')
        regions_data = {}
        output_data = {}

        for i in range (0,fcn.shape[0]):
            rect = (int(fcn[i,0]), int(fcn[i,1]), int(fcn[i,2]), int(fcn[i,3]))
            regions_data[rect] = fcn[i, 4] #FCN score
            output_data[rect] = 0 #FCN score

        weak = np.genfromtxt(file, delimiter=',')
        weak[:, 4] = weak[:,4] * -1

        for i in range(0, weak.shape[0]):
            rect = (int(weak[i,0]), int(weak[i,1]), int(weak[i,2]), int(weak[i,3]))
            if regions_data[rect] > coef and weak[i,4] > output_data[rect]: #If is above energy TH and has not grater value keep weak class score
                output_data[rect] = weak[i,4]

        i = 0
        for key, value in output_data.iteritems():
            fcn[i,0] = int(key[0])
            fcn[i,1] = int(key[1])
            fcn[i,2] = int(key[2])
            fcn[i,3] = int(key[3])
            fcn[i,4] = value
            i += 1

        fcn = fcn[fcn[:, 4].argsort()[::-1]] #Top ranked go first

        # Create dir and save data
        directory = '../blabla' +database+'/conf_' +net+ '_suppression_0' + str(coef)[2:] + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savetxt(directory + file.split('/')[index-1], fcn, delimiter=",")

        print file.split('/')[index-1]

    if count == 0:
        print 'No files found in that folder'
    else:
        print 'Done for ' + str(count) + ' files.'

#Performs linear combination of scores. Args: Name of the net, coef of combination (multiplied by weak class)
def linear_fusion(database, net, coef):

    #coef = 0.5 #Multiplied by weak classifier
    coef = float(coef)
    count = 0
    for file in glob.glob('../blabla' +database+'/proposals/*.csv'):
        count += 1

        index = file.split('/').__len__()
        fcn = np.genfromtxt('../blabla' +database+'/conf_' +net+ '/' + file.split('/')[index-1], delimiter=',')
        regions_data = {}
        out_data = {}
        aux_data = {}


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
        for key, value in out_data.iteritems():
            fcn[i,0] = int(key[0])
            fcn[i,1] = int(key[1])
            fcn[i,2] = int(key[2])
            fcn[i,3] = int(key[3])
            fcn[i,4] = value
            i += 1

        fcn = fcn[fcn[:, 4].argsort()[::-1]] #Top ranked go first

        # Create dir and save data
        directory = '../blabla' +database+'/conf_' + net + '_fusion_0' + str(coef)[2:] + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savetxt(directory + file.split('/')[index - 1], fcn, delimiter=",")

        print file.split('/')[index - 1]

    if count == 0:
        print 'No files found in that folder'
    else:
        print 'Done for ' + str(count) + ' files.'


if __name__ == '__main__':
    params = [(len(p) > 0 and p[0] != '-', p) for p in sys.argv]
    sys.argv = [p[1] for p in params if p[0]]

    if sys.argv[1] == 'suppression':
        suppression(sys.argv[2], sys.argv[3], sys.argv[4])

    elif sys.argv[1] == 'linear':
        linear_fusion(sys.argv[2], sys.argv[3], sys.argv[4])

    else:
        print 'Wrong function'

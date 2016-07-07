import numpy as np
from numpy import genfromtxt

va = genfromtxt('va.txt', delimiter=' ')
vb = genfromtxt('v3.csv', delimiter=',')

diff = va[:,5] - vb[:,4]
proportion =  va[:,5] / vb[:,4]

area_prop = [va[:,2] * va[:,3], proportion]
#area_prop[np.argsort(area_prop[:, 0])]

print sum(np.absolute(diff)) / diff.__len__()

print 'Done'
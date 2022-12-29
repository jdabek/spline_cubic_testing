import numpy as np
import matplotlib.pyplot as plt

# Implementation of Cubic Spline, based on:
# https://en.wikipedia.org/wiki/Spline_(mathematics)
# as on 2022-12-29

def verifyData(x, y):
    if len(x) < 2 or len(x) != len(y):
        raise ValueError('Incompatible data')
    for i in range(1,len(x)):
        if x[i] <= x[i-1]:
            raise ValueError('Data not strictly monotonic')

def getSplineData(xvec, yvec):
    verifyData(xvec, yvec)
    
    ninterp = len(xvec)-1
    
    hvec = [0.0]*ninterp
    for i in range(ninterp):
        hvec[i] = xvec[i+1]-xvec[i]
    
    avec = [0.0]*ninterp
    for i in range(1,ninterp):
        avec[i] = 3.0*((yvec[i+1]-yvec[i])/hvec[i] - (yvec[i]-yvec[i-1])/hvec[i-1])
    
    lvec = [1.0]*ninterp
    muvec = [0.0]*ninterp
    zvec = [0.0]*ninterp
    for i in range(1,ninterp):
        lvec[i] = 2.0*(xvec[i+1]-xvec[i-1])-hvec[i-1]*muvec[i-1]
        muvec[i] = hvec[i]/lvec[i]
        zvec[i] = (avec[i] - hvec[i-1]*zvec[i-1])/lvec[i]
    
    cvec = [0.0]*(ninterp+1)
    for i in range(ninterp-1,0,-1):
        cvec[i] = zvec[i] - muvec[i]*cvec[i+1]

    splineData = {}

    splineData['zvec'] = [2.0*c for c in cvec]
    splineData['hvec'] = hvec
    splineData['xvec'] = xvec
    splineData['yvec'] = yvec

    return splineData

def getSplineValues(xinterp, splineData):
    ninterp = len(xinterp)
    yinterp = [0.0]*ninterp
    for i in range(ninterp):
        if xinterp[i] < splineData['xvec'][0] or xinterp[i] > splineData['xvec'][-1]:
            raise ValueError('xinterp out of interpolation range')
        for j in range(1,len(splineData['xvec'])):
            if xinterp[i] <= splineData['xvec'][j]:
                yinterp[i] = (splineData['zvec'][j]*(xinterp[i]-splineData['xvec'][j-1])**3  \
                           + splineData['zvec'][j-1]*(splineData['xvec'][j]-xinterp[i])**3)/(6.0*splineData['hvec'][j-1]) \
                           + (splineData['yvec'][j]/splineData['hvec'][j-1]-splineData['zvec'][j]*splineData['hvec'][j-1]/6.0)*(xinterp[i]-splineData['xvec'][j-1]) \
                           + (splineData['yvec'][j-1]/splineData['hvec'][j-1]-splineData['zvec'][j-1]*splineData['hvec'][j-1]/6.0)*(splineData['xvec'][j]-xinterp[i])
                break

    return yinterp

nsamp = 20
x = np.random.rand(nsamp)
x = np.sort(x)
y = np.random.rand(nsamp)

splineData = getSplineData(x, y)
ninterp = 1000
xinterp = [x[0]+i*(x[-1]-x[0])/ninterp for i in range(ninterp+1)]
yinterp = getSplineValues(xinterp, splineData)

plt.plot(x, y, 'x')
plt.plot(xinterp, yinterp)

plt.savefig('spline_example.png')

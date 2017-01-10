from __future__ import division
from libc.math cimport sqrt
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def fillWeights(np.ndarray[DTYPE_t, ndim=2] movieUserRatings,np.ndarray[DTYPE_t, ndim=2] userRateMeans,int userSize, int movieSize):

    cdef int i,j
    cdef float temp,totUpper,totBottomActive,totBottomOther,activeUsr,otherUsr
    cdef np.ndarray[DTYPE_t, ndim=2] weightMatrix = np.zeros([userSize,userSize],dtype = DTYPE)
    for i in range(0,userSize):
        if (i % 1000) == 0:
            print(str(i) + 'th user')
        for j in range((i+1),userSize):
            totUpper = 0
            totBottomActive = 0
            totBottomOther = 0
            for k in range(0,movieSize):
                activeUsr = movieUserRatings[k,i]
                otherUsr = movieUserRatings[k,j]
                if activeUsr != 0 and otherUsr != 0:
                    totUpper += (activeUsr - userRateMeans[0,i]) * (otherUsr - userRateMeans[0,j])
                    # did not used pow() for computational strain
                    totBottomActive += (activeUsr - userRateMeans[0,i]) * (activeUsr - userRateMeans[0,i])
                    totBottomOther += (otherUsr - userRateMeans[0,j]) * (otherUsr - userRateMeans[0,j])
            temp = sqrt(totBottomActive * totBottomOther)
            if temp != 0:
                weightMatrix[i][j] = totUpper / temp
                weightMatrix[j][i] = weightMatrix[i][j]
    return weightMatrix

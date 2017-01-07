from __future__ import division
import numpy as np
import pandas
import datetime
import math

columnNames = ['movieID', 'userID', 'rating']
rawData = pandas.read_csv('TrainingRatings.txt', names=columnNames,
                        dtype={'movieID': np.str, 'userID': np.str, 'rating': np.float})


uniqueMovieID = rawData.movieID.unique()
uniqueUserID = rawData.userID.unique()

#creating index dictionary for movie and user
a = dict(enumerate(uniqueMovieID))
movieDict = {v: k for k, v in a.iteritems()}
a= dict(enumerate(uniqueUserID))
userDict = {v: k for k, v in a.iteritems()}


movieUserRatings = np.zeros([uniqueMovieID.size,uniqueUserID.size], dtype = np.float)
userRateMeans = np.zeros([1,uniqueUserID.size],dtype = np.float)
weightMatrix = np.zeros([uniqueUserID.size,uniqueUserID.size],dtype = np.float)
# filling matrix with rating info

for datInd,datRow in rawData.iterrows():
    movInd = movieDict[datRow[columnNames[0]]]
    usrInd = userDict[datRow[columnNames[1]]]
    movieUserRatings[movInd][usrInd] = datRow[columnNames[2]]

#taking mean of uservotes discarding unvoted films
for i in range(0,uniqueUserID.size):
    count = 0
    tot = 0
    for j in range(0,uniqueMovieID.size):
        if movieUserRatings[j][i] != 0:
            count += 1
            tot += movieUserRatings[j][i]
    userRateMeans[0][i] = tot / count

#filling the weight matrix
print('filling weight matrix')
print (datetime.datetime.now())

for i in range(0,uniqueUserID.size):
    for j in range(0,uniqueUserID.size):
        totUpper = 0
        totBottomActive = 0
        totBottomOther = 0
        for k in range(0,uniqueMovieID.size):
            activeUsr = movieUserRatings[k][i]
            otherUsr = movieUserRatings[k][j]
            if activeUsr != 0 and otherUsr != 0:
                totUpper += (activeUsr - userRateMeans[0][i]) * (otherUsr - userRateMeans[0][j])
                # did not used pow() for computational strain
                totBottomActive += (activeUsr - userRateMeans[0][i]) * (activeUsr - userRateMeans[0][i])
                totBottomOther += (otherUsr - userRateMeans[0][j]) * (otherUsr - userRateMeans[0][j])
        temp = math.sqrt(totBottomActive * totBottomOther)
        if temp != 0:
            weightMatrix[i][j] = totUpper / temp

print('weighting is done')
print (datetime.datetime.now())

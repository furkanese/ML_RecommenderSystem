from __future__ import division
import numpy as np
import pandas
import datetime
import math

columnNames = ['movieID', 'userID', 'rating']

print('reading data')
print (datetime.datetime.now())

rawData = pandas.read_csv('TrainingRatings.txt', names=columnNames,
                        dtype={'movieID': np.str, 'userID': np.str, 'rating': np.float})

testRawData = pandas.read_csv('TestingRatings.txt', names=columnNames,
                        dtype={'movieID': np.str, 'userID': np.str, 'rating': np.float})

print('data read. parsing')
print (datetime.datetime.now())
uniqueMovieID = rawData.movieID.unique()
uniqueUserID = rawData.userID.unique()

testUniqueMovieID = testRawData.movieID.unique()
testUniqueUserID = testRawData.userID.unique()

#creating index dictionary for movie and user
#training
a = dict(enumerate(uniqueMovieID))
movieDict = {v: k for k, v in a.iteritems()}
a= dict(enumerate(uniqueUserID))
userDict = {v: k for k, v in a.iteritems()}
#test
a = dict(enumerate(testUniqueMovieID))
testMovieDict = {v: k for k, v in a.iteritems()}
a= dict(enumerate(testUniqueUserID))
testUserDict = {v: k for k, v in a.iteritems()}
print('Parsing done. Putting test and train data into matrix and taking means')
print (datetime.datetime.now())

movieUserRatings = np.zeros([uniqueMovieID.size,uniqueUserID.size], dtype = np.float)
userRateMeans = np.zeros([1,uniqueUserID.size],dtype = np.float)

testMovieUserRatings = np.zeros([testUniqueMovieID.size,testUniqueUserID.size],dtype = np.float)

weightMatrix = np.zeros([uniqueUserID.size,uniqueUserID.size],dtype = np.float)

def fillMatrix(rawData,movieUserRate,movieDic,userDic):
    # filling matrix with rating info
    for datInd,datRow in rawData.iterrows():
        movInd = movieDic[datRow[columnNames[0]]]
        usrInd = userDic[datRow[columnNames[1]]]
        movieUserRate[movInd][usrInd] = datRow[columnNames[2]]

    return movieUserRate

def calcMeans(movieUserRate,userRateMean,userSize,movieSize):

    # taking mean of user votes discarding unvoted films
    for i in range(0,userSize):
        count = 0
        tot = 0
        for j in range(0,movieSize):
            if movieUserRate[j][i] != 0:
                count += 1
                tot += movieUserRatings[j][i]
        userRateMean[0][i] = tot / count
    return userRateMean


#train
movieUserRatings = fillMatrix(rawData,movieUserRatings,movieDict,userDict)
userRateMeans = calcMeans(movieUserRatings,userRateMeans,uniqueUserID.size,uniqueMovieID.size)
#test
testMovieUserRatings = fillMatrix(testRawData,testMovieUserRatings,testMovieDict,testUserDict)

#filling the weight matrix
print('Done. Filling weight matrix')
print (datetime.datetime.now())

for i in range(0,uniqueUserID.size):
    if (i % 100) == 0:
        print(str(i) + 'th user')
    for j in range((i+1),uniqueUserID.size):
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
            weightMatrix[j][i] = weightMatrix[i][j]

print('weighting is done')
print (datetime.datetime.now())
#print(weightMatrix[:10,:10])


'''
np.savetxt('weights.csv',weightMatrix,delimiter=',')
print('saved the weights')
print (datetime.datetime.now())
'''

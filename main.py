from __future__ import division
import numpy as np
import pandas
import datetime
import math
import fillweights

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
print(uniqueMovieID.size)
print (uniqueUserID.size)

testUniqueMovieID = testRawData.movieID.unique()
testUniqueUserID = testRawData.userID.unique()

print(testUniqueMovieID.size)
print (testUniqueUserID.size)

#creating index dictionary for movie and user
#training
indexToMovie = dict(enumerate(uniqueMovieID))
movieDict = {v: k for k, v in indexToMovie.iteritems()}
indexToUser = dict(enumerate(uniqueUserID))
userDict = {v: k for k, v in indexToUser.iteritems()}
#test
testIndexToMovie = dict(enumerate(testUniqueMovieID))
testMovieDict = {v: k for k, v in testIndexToMovie.iteritems()}
testIndexToUser = dict(enumerate(testUniqueUserID))
testUserDict = {v: k for k, v in testIndexToUser.iteritems()}
print('Parsing done. Putting test and train data into matrix and taking means')
print (datetime.datetime.now())

movieUserRatings = np.zeros([uniqueMovieID.size,uniqueUserID.size], dtype = np.float)
userRateMeans = np.zeros([1,uniqueUserID.size],dtype = np.float)

testMovieUserRatings = np.zeros([testUniqueMovieID.size,testUniqueUserID.size],dtype = np.float)

#weightMatrix = np.zeros([uniqueUserID.size,uniqueUserID.size],dtype = np.float)

def fillMatrix(rawData,movieUserRate,movieDic,userDic):
    # filling matrix with rating info
    totVoteCount = 0
    for datInd,datRow in rawData.iterrows():
        movInd = movieDic[datRow[columnNames[0]]]
        usrInd = userDic[datRow[columnNames[1]]]
        movieUserRate[movInd][usrInd] = datRow[columnNames[2]]
        totVoteCount += 1
    return movieUserRate,totVoteCount

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
testMovieUserRatings,testTotalVote = fillMatrix(testRawData,testMovieUserRatings,testMovieDict,testUserDict)
#filling the weight matrix
print('Done. Filling weight matrix')
print (datetime.datetime.now())

weightMatrix = fillweights.fillWeights(movieUserRatings,userRateMeans,uniqueUserID.size,uniqueMovieID.size)

print('weighting is done')
print (datetime.datetime.now())
#print(weightMatrix[:10,:10])

print('taking predictions')

currentUser = 0
currentMovie = 0
totUp = 0
totDown = 0
neighbourSize = 3

prediction = np.zeros([testTotalVote], dtype = [('movName','|S10'),('userName','|S10'),('pred','f4')])

cntr = 0
for j in range(0,testUniqueUserID.size):
    currentUser = userDict[testIndexToUser[j]]
    sortWeights = np.argsort(weightMatrix[currentUser][:])
    for i in range(0,testUniqueMovieID.size):
        currentMovie = movieDict[testIndexToMovie[i]]
        for k in range(0,neighbourSize):
            #taking most similar users
            mostSim = int(sortWeights[k])
            totUp += (movieUserRatings[i][mostSim] - userRateMeans[0][mostSim]) * weightMatrix[currentUser][mostSim]
            totDown += weightMatrix[currentUser][mostSim]
        if totDown != 0:
            pred = userRateMeans[0][currentUser] + (totUp / totDown)
        else:
            pred = userRateMeans[0][currentUser]
        a = ('%f' % testIndexToMovie[i]).rstrip('0').rstrip('.')
        b = ('%f' % testIndexToUser[i]).rstrip('0').rstrip('.')
        prediction[cntr] = (a,b,pred)
        '''
        prediction[cntr][0] = testIndexToMovie[i]
        prediction[cntr][1] = testIndexToUser[j]
        if totDown > 0:
            prediction[cntr][2] = userRateMeans[0][currentUser] + (totUp / totDown)
        else:
            prediction[cntr][2] = userRateMeans[0][currentUser]
        '''
        totDown = 0
        totUp = 0
        cntr += 1


np.savetxt('predictions.txt',prediction,delimiter=',')
print('saved the prediction')
print (datetime.datetime.now())



'''
np.savetxt('weights.csv',weightMatrix,delimiter=',')
print('saved the weights')
print (datetime.datetime.now())
'''

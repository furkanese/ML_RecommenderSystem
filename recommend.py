from __future__ import division
import numpy as np
import pandas
import datetime
import math
import fillweights

columnNames = ['movieID', 'userID', 'rating']

print('reading data')
print (datetime.datetime.now())

testRawData = pandas.read_csv('TestingRatings.txt', names=columnNames,
                        dtype={'movieID': np.str, 'userID': np.str, 'rating': np.float})

predictRawData = pandas.read_csv('predictions.txt', names=columnNames,
                        dtype={'movieID': np.str, 'userID': np.str, 'rating': np.float})

print('data read. parsing')
print (datetime.datetime.now())
testUniqueMovieID = testRawData.movieID.unique()
testUniqueUserID = testRawData.userID.unique()
print(testUniqueMovieID.size)
print (testUniqueUserID.size)

predictUniqueMovieID = predictRawData.movieID.unique()
predictUniqueUserID = predictRawData.userID.unique()

print(predictUniqueMovieID.size)
print (predictUniqueUserID.size)

#creating index dictionary for movie and user
#test
testIndexToMovie = dict(enumerate(testUniqueMovieID))
testMovieDict = {v: k for k, v in testIndexToMovie.iteritems()}
testIndexToUser = dict(enumerate(testUniqueUserID))
testUserDict = {v: k for k, v in testIndexToUser.iteritems()}
#predict
predictIndexToMovie = dict(enumerate(predictUniqueMovieID))
predictMovieDict = {v: k for k, v in predictIndexToMovie.iteritems()}
predictIndexToUser = dict(enumerate(predictUniqueUserID))
predictUserDict = {v: k for k, v in predictIndexToUser.iteritems()}
print('Parsing done. Putting votes to matrix')
print (datetime.datetime.now())

# test
testMovieUserRatings = np.zeros([testUniqueMovieID.size,testUniqueUserID.size], dtype = np.float)

# predict
predictMovieUserRatings = np.zeros([predictUniqueMovieID.size,predictUniqueUserID.size],dtype = np.float)


def fillMatrix(rawData,movieUserRate,movieDic,userDic):
    # filling matrix with rating info
    totVoteCount = 0
    for datInd,datRow in rawData.iterrows():
        movInd = movieDic[datRow[columnNames[0]]]
        usrInd = userDic[datRow[columnNames[1]]]
        movieUserRate[movInd][usrInd] = datRow[columnNames[2]]
        totVoteCount += 1
    return movieUserRate,totVoteCount

# testM
testMovieUserRatings,testTotalVote = fillMatrix(testRawData,testMovieUserRatings,testMovieDict,testUserDict)

# predict
predictMovieUserRatings,predictTotalVote = fillMatrix(predictRawData,predictMovieUserRatings,predictMovieDict,predictUserDict)

print (testTotalVote,predictTotalVote)
print('Done. Taking error.')
print (datetime.datetime.now())

totError = 0
cntr = 0
biggerThan4 = 0
totRmse = 0

for i in range (0,testUniqueMovieID.size):
    predMovInd = predictMovieDict[testIndexToMovie[i]]
    for j in range(0,testUniqueUserID.size):
        predUsrInd = predictUserDict[testIndexToUser[j]]
        if testMovieUserRatings[i][j] != 0 and predictMovieUserRatings[predMovInd][predUsrInd] != 0:
            totError += abs(predictMovieUserRatings[predMovInd][predUsrInd] - testMovieUserRatings[i][j])
            totRmse += (predictMovieUserRatings[predMovInd][predUsrInd] - testMovieUserRatings[i][j]) * (predictMovieUserRatings[predMovInd][predUsrInd] - testMovieUserRatings[i][j])
            cntr += 1
        if predictMovieUserRatings[predMovInd][predUsrInd] >= 4:
            biggerThan4 += 1

#print(totError)
#print(cntr)

print('Mean Absolute Error: ')
print(totError / cntr)
print('RMSE')
print(math.sqrt(totRmse / cntr))

print('Predictions bigger than 4: ' , str(biggerThan4))
cntr = 0
prediction = np.zeros([biggerThan4], dtype = [('movName','i'),('userName','|S10')])
for i in range (0,predictUniqueMovieID.size):
    predMov = predictIndexToMovie[i]
    for j in range(0,predictUniqueUserID.size):
        predUsr = predictIndexToUser[j]
        if predictMovieUserRatings[i][j] >= 4:
            prediction[cntr] = (predMov,predUsr)
            cntr += 1

# sorting by movie
inds = np.argsort(prediction['movName'])
np.take(prediction,inds,out = prediction)
np.savetxt('recommendations.txt',prediction,delimiter=',',fmt = '%d,%s')
print('saved the prediction')
print (datetime.datetime.now())

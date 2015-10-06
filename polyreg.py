'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        self.degree = degree
        self.regLambda = regLambda
        self.theta = None


    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        #TODO
        expandedArr = []
        for x in range(0, X.size):
            curArr = []
            for y in range(0, degree):
                curArr.append(X[x]**(1 + y))
            expandedArr.append(curArr)
        return expandedArr

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''
        # convert X into a n * d array of polynomial features of 
        # degree d
        XExpanded = self.polyfeatures(X, self.degree)

        # add the zero-th order feature row (i.e. x_0 = 1)
        for x in range(0, X.size):
            XExpanded[x].insert(0, 1)

        XExpandedNP = np.array(XExpanded)
        n,d = XExpandedNP.shape
        XExpandedNPCopy = np.zeros((n,d))
        for x in range(0, n):
            for z in range(0, d):
                XExpandedNPCopy[x][z] = XExpandedNP[x][z]

        # standardize the data before fitting
        for x in range(0, X.size):
            for deg in range(1, self.degree + 1):
                curCol = XExpandedNPCopy[:,deg]
                curColMean = curCol.mean()
                curColStd = curCol.std()
                newVal = (XExpandedNP[x][deg] - curColMean) / curColStd
                XExpandedNP[x][deg] = newVal
        # fit
        n,d = XExpandedNP.shape
        d = d - 1
        regMatrix = self.regLambda * np.eye(d + 1)
        regMatrix[0,0] = 0

        self.theta = np.linalg.pinv(XExpandedNP.T.dot(XExpandedNP) + regMatrix).dot(XExpandedNP.T).dot(y);
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''       

        XExpanded = self.polyfeatures(X, self.degree)

        # add the zero-th order feature row (i.e. x_0 = 1)
        for x in range(0, X.size):
            XExpanded[x].insert(0, 1)

        XExpandedNP = np.array(XExpanded)
        n,d = XExpandedNP.shape
        XExpandedNPCopy = np.zeros((n,d))
        for x in range(0, n):
            for z in range(0, d):
                XExpandedNPCopy[x][z] = XExpandedNP[x][z]

        # standardize the data before fitting
        if len(X) > 1:
            for x in range(0, X.size):
                for deg in range(1, self.degree + 1):
                    curCol = XExpandedNPCopy[:,deg]
                    curColMean = curCol.mean()
                    curColStd = curCol.std()
                    newVal = (XExpandedNP[x][deg] - curColMean) / curColStd
                    XExpandedNP[x][deg] = newVal

        return XExpandedNP.dot(self.theta)

#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain);
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    
    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape
    
    modelTrain = PolynomialRegression(degree = degree, regLambda = regLambda)

    # training
    i = 2
    while i < n:
        modelTrain.fit(Xtrain[0:i], Ytrain[0:i])
        predictions = modelTrain.predict(Xtrain[0:i])

        cumSumError = 0
        t = 0
        while t < i:
            cumSumError += (predictions[t] - Ytrain[t])**2
            t += 1
        
        cumSumError = cumSumError / n
        errorTrain[i] = cumSumError
        i += 1

    modelTest = PolynomialRegression(degree = degree, regLambda = regLambda)

    # testing
    #standardize Xtest based on the training mean and standard deviation
    for i in range(0, len(Xtest)):
        Xtest[i] = (Xtest[i] - Xtrain.mean())/(Xtrain.std())

    i = 2
    while i < n:
        modelTest.fit(Xtrain[0:i], Ytrain[0:i])

        predictions = modelTest.predict(Xtest)
        cumSumError = 0

        t = 0
        while t < len(predictions):
            cumSumError += (predictions[t] - Ytest[t])**2
            t += 1
        
        cumSumError = cumSumError / n
        errorTest[i] = cumSumError
        i += 1

    return (errorTrain, errorTest)

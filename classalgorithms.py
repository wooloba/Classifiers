from __future__ import division  # floating point division
import numpy as np
import utilities as utils

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it should ignore this last feature
        self.params = {'usecolumnones': True}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.means = []
        self.stds = []
        self.numfeatures = 0
        self.numclasses = 0

        #P(y=0) & P(y=1)
        self.py0 = 0.0
        self.py1 = 0.0
    def learn(self, Xtrain, ytrain):
        """
        In the first code block, you should set sample_num and
        self.numfeatures correctly based on the inputs and the given parameters
        (use the column of ones or not).

        In the second code block, you should compute the parameters for each
        feature. In this case, they're mean and std for Gaussian distribution.
        """

        ### YOUR CODE HERE
        self.numclasses = 2
        sample_num = Xtrain.shape[0]
        if self.params['usecolumnones']:
            self.numfeatures = Xtrain.shape[1]
        else:
            self.numfeatures = Xtrain.shape[1]-1


        self.py0 = (sample_num-sum(ytrain))/sample_num
        self.py1 = (sum(ytrain))/sample_num

        ### END YOUR CODE

        origin_shape = (self.numclasses, self.numfeatures)
        self.means = np.zeros(origin_shape)
        self.stds = np.zeros(origin_shape)

        ### YOUR CODE HERE
        # mu j,c
        for i in range(self.numfeatures):
            feature_mean_c1 = 0
            feature_mean_c0 = 0
            for j in range(sample_num):
                #class is 1
                if ytrain[j] == 1:
                    feature_mean_c1 += Xtrain[j,i]
                #class is 0
                else:
                    feature_mean_c0 += Xtrain[j,i]
            #mu j,c. Where j is feature(8/9), c is class{0,1}
            # 0 -> class = 0; 1-> class = 1

            self.means[0][i] = feature_mean_c0/(sample_num-sum(ytrain))
            self.means[1][i] = feature_mean_c1/(sum(ytrain))

        #sigma j,c
        for i in range(self.numfeatures):
            feature_sigma_c1 = 0
            feature_sigma_c0 = 0
            for j in range(sample_num):
                #class is 1
                if ytrain[j] == 1:
                    feature_sigma_c1 += (Xtrain[j,i] - self.means[0][i])**2
                #class is 0
                else:
                    feature_sigma_c0 += (Xtrain[j,i] - self.means[0][i])**2
            #sigma j,c
            # 0 -> class = 0; 1-> class = 1
            self.stds[0][i] = feature_sigma_c0/(sample_num-sum(ytrain))
            self.stds[1][i] = feature_sigma_c1/(sum(ytrain))
        ### END YOUR CODE


        assert self.means.shape == origin_shape
        assert self.stds.shape == origin_shape

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        sample_num = Xtest.shape[0]
        ### YOUR CODE HERE
        for i in range(sample_num):
            pxy_c1 = 1.0
            pxy_c0 = 1.0

            for j in range(self.numfeatures):
                pxy_c0 *= (1 / np.sqrt(2 * np.pi * self.stds[0][j])) * np.exp(-np.square(Xtest[i][j] - self.means[0][j]) / (2 * self.stds[0][j]))
                pxy_c1 *= (1 / np.sqrt(2 * np.pi * self.stds[1][j])) * np.exp(-np.square(Xtest[i][j] - self.means[1][j]) / (2 * self.stds[1][j]))

            ytest[i] = np.argmax([pxy_c0*self.py0,pxy_c1*self.py1])

        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]

        return ytest

class LogitReg(Classifier):

    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """

        cost = 0.0

        ### YOUR CODE HERE

        ### END YOUR CODE

        return cost

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """

        grad = np.zeros(len(theta))

        ### YOUR CODE HERE

        ### END YOUR CODE

        return grad

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """
        #Xtrain = np.delete(Xtrain,8,axis=1)
        self.weights = np.zeros(Xtrain.shape[1],)
        ### YOUR CODE HERE
        learning_rate = self.params['regwgt']
        iter = 0

        while iter < 25:
            for i in range(ytrain.shape[0]):
                self.weights = self.weights - learning_rate*np.dot(utils.sigmoid(np.dot(Xtrain[i],self.weights))-ytrain[i],Xtrain[i])

            iter +=1
        ### END YOUR CODE

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        #Xtest = np.delete(Xtest, 8, axis=1)
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        ytest = utils.sigmoid(np.dot(self.weights,Xtest.T))

        ytest[ytest >= 0.5] = 1
        ytest[ytest < 0.5] = 0

        ### END YOUR CODE
        assert len(ytest) == Xtest.shape[0]
        return ytest

class NeuralNet(Classifier):
    """ Implement a neural network with a single hidden layer. Cross entropy is
    used as the cost function.

    Parameters:
    nh -- number of hidden units
    transfer -- transfer function, in this case, sigmoid
    stepsize -- stepsize for gradient descent
    epochs -- learning epochs

    Note:
    1) feedforword will be useful! Make sure it can run properly.
    2) Implement the back-propagation algorithm with one layer in ``backprop`` without
    any other technique or trick or regularization. However, you can implement
    whatever you want outside ``backprob``.
    3) Set the best params you find as the default params. The performance with
    the default params will affect the points you get.
    """
    def __init__(self, parameters={}):
        self.params = {'nh': 16,
                    'transfer': 'sigmoid',
                    'stepsize': 0.01,
                    'epochs': 10}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.w_input = None
        self.w_output = None

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        # hidden activations
        a_hidden = self.transfer(np.dot(inputs,self.w_input))
        #print(self.w_input.shape,inputs.shape,a_hidden.shape) #---->((nh, 9), (9,), (nh,))

        # output activations
        a_output = self.transfer(np.dot(a_hidden, self.w_output))
        #print(self.w_output.shape,a_hidden.shape,a_output.shape) # ------>((1, nh), (nh,), (1,))

        return (a_hidden, a_output)

    def backprop(self, x,y):
        """
        Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """

        ### YOUR CODE HERE

        ### END YOUR CODE
        # assert nabla_input.shape == self.w_input.shape
        # assert nabla_output.shape == self.w_output.shape
        # return (nabla_input, nabla_output)

    def learn(self, Xtrain, ytrain):

        print(self.params)

        # W(1) 9xnh
        self.w_input = np.random.normal(0.0,1.0,(Xtrain.shape[1], self.params["nh"]))

        # W(2) nhx1
        self.w_output = np.random.normal(0.0,1.0,(self.params["nh"],1))

        iter = 0
        while iter <= self.params['epochs']:
            #1. forward propagate, ao---> predict,y_hat
            ah,ao = self.feedforward(Xtrain)

            # back_probagation
            error = np.reshape(ytrain,[ytrain.shape[0],1] ) - ao

            d_W_output = error * utils.dsigmoid(ao)
            d_W_output = np.dot(ah.T,d_W_output)


            d_W_input = np.dot(error * utils.dsigmoid(ao),self.w_output.T)
            d_W_input = d_W_input *utils.dsigmoid(ah)

            d_W_input = np.dot(Xtrain.T,d_W_input)

            self.w_output += self.params['stepsize'] * d_W_output
            self.w_input += self.params['stepsize'] * d_W_input

            iter += 1
        return

    def predict(self, Xtest):
        #Xtest = np.delete(Xtest, 8, axis=1)
        ytest = np.zeros(Xtest.shape[0])
        for i in range(len(Xtest)):
            ah,ao = self.feedforward(Xtest[i])
            if ao >= 0.5:
                ytest[i] = 1
            else:
                ytest[i] = 0
        return ytest

    # TODO: implement learn and predict functions

class KernelLogitReg(LogitReg):
    """ Implement kernel logistic regression.

    This class should be quite similar to class LogitReg except one more parameter
    'kernel'. You should use this parameter to decide which kernel to use (None,
    linear or hamming).

    Note:
    1) Please use 'linear' and 'hamming' as the input of the paramteter
    'kernel'. For example, you can create a logistic regression classifier with
    linear kerenl with "KernelLogitReg({'kernel': 'linear'})".
    2) Please don't introduce any randomness when computing the kernel representation.
    """
    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None', 'kernel': 'None'}
        self.reset(parameters)


    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data.

        Ktrain the is the kernel representation of the Xtrain.
        """
        self.Xtrain = Xtrain
        self.center = Xtrain[:20]

        ### YOUR CODE HERE
        if self.params['kernel'] == 'linear':
            Ktrain = np.dot(Xtrain, self.center.T)
        elif self.params['kernel'] == 'hamming':
            Ktrain = np.zeros(shape=(Xtrain.shape[0],len(self.center)))
            for i in range(Ktrain.shape[0]):
                for j in range(Ktrain.shape[1]):
                    Ktrain[i, j] = self.hammingDistance(Xtrain[i], self.center[j])

        elif self.params['kernel'] == 'None':
            Ktrain = Xtrain

        ### END YOUR CODE
        self.weights = np.zeros(Ktrain.shape[1],)
        ### YOUR CODE HERE
        learning_rate = self.params['regwgt']
        iter = 0
        while iter < 50:
            for i in range(ytrain.shape[0]):
                self.weights = self.weights - learning_rate * np.dot(utils.sigmoid(np.dot(Ktrain[i], self.weights)) - ytrain[i], Ktrain[i])

            iter += 1
        ### END YOUR CODE
        self.transformed = Ktrain # Don't delete this line. It's for evaluation.

    # TODO: implement necessary functions
    def predict(self, Xtest):
        if self.params['kernel'] == 'linear':
            Ktest = np.dot(Xtest, self.center.T)
        elif self.params['kernel'] == 'hamming':
            Ktest = np.zeros((Xtest.shape[0], len(self.center)))

            for i in range(Ktest.shape[0]):
                for j in range(Ktest.shape[1]):
                    Ktest[i][j] = self.hammingDistance(Xtest[i], self.center[j])

        elif self.params['kernel'] == 'None':
            Ktest = Xtest

        ytest = utils.sigmoid(np.dot(self.weights,Ktest.T ))

        ytest[ytest >= 0.5] = 1
        ytest[ytest < 0.5] = 0

        ### END YOUR CODE
        assert len(ytest) == Xtest.shape[0]
        return ytest


    def hammingDistance(self,a,b):
        dis = 0
        for i in range(len(a)):
            if a[i] == b[i]:
                dis += 1

        return dis

# ======================================================================

def test_lr():
    print("Basic test for logistic regression...")
    clf = LogitReg()
    theta = np.array([0.])
    X = np.array([[1.]])
    y = np.array([0])

    try:
        cost = clf.logit_cost(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost!")
    assert isinstance(cost, float), "logit_cost should return a float!"

    try:
        grad = clf.logit_cost_grad(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost_grad!")
    assert isinstance(grad, np.ndarray), "logit_cost_grad should return a numpy array!"

    print("Test passed!")
    print("-" * 50)

def test_nn():
    print("Basic test for neural network...")
    clf = NeuralNet()
    X = np.array([[1., 2.], [2., 1.]])
    y = np.array([0, 1])
    clf.learn(X, y)

    assert isinstance(clf.w_input, np.ndarray), "w_input should be a numpy array!"
    assert isinstance(clf.w_output, np.ndarray), "w_output should be a numpy array!"

    try:
        res = clf.feedforward(X[0, :])
    except:
        raise AssertionError("feedforward doesn't work!")

    try:
        res = clf.backprop(X[0, :], y[0])
    except:
        raise AssertionError("backprob doesn't work!")

    print("Test passed!")
    print("-" * 50)

def main():
    test_lr()
    test_nn()

if __name__ == "__main__":
    main()

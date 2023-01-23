import numpy as np


class LogisticRegression:

    def __init__(self, x_dim):
        self.theta = np.zeros(x_dim)
        self.cost_history = []
        self.theta_history = []

    def predict(self, X):
        """
        Computes the prediction (hzpothesis) of the linear regression
        :param X: input data as row vectors
        :return: vector of predicted outputs
        """
        # TODO
        y = []
        for i in range(len(X)):
            p = self.get_positive_score(X[i])
            if p > 0:
                y.append(1)
            else:
                y.append(0)
        return np.asarray(y)

    def get_positive_score(self, X):
        """
        Computes the probability of classification to the positive class
        :param X: Input data
        :return:
        """
        # TODO
        return (1/(1 + np.exp(-np.dot(self.theta, X.T))))

    def cost(self, X, y):
        """
        Computes the loss function of a linear regression (mean square error)
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return: Loss value
        """
        # TODO
        c = 0
        for i in range(len(X)):
            c = -y[i] * np.log(self.get_positive_score(X[i])) - (1 - y[i])*np.log(1- self.get_positive_score(X[i]))
        return c/len(X)

    def grad(self, X, y):
        """
        Computes the gradient of the loss function with regard to the parameters theta
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return: Gradient
        """
        gsol = []
        for i in range(len(self.theta)):
            aux = 0
            for j in range(X.shape[0]):
                aux += X[j][i] * (self.get_positive_score(X[j])) - y[j] * X[j][i]
            aux = aux/X.shape[0] 
            gsol.append(aux)
        return gsol

    def update(self, theta, cost):
        # print("%s : grad = %s, cost = %s" % (str(self.theta), str(G), str(self.__cost)))
        self.theta = theta
        self.theta_history.append(np.copy(self.theta))
        self.cost_history.append(cost)


class OneVsAll:

    def __init__(self, model_gen, opt_gen):
        """
        One-vs-all technique implementation
        :param model_gen: a generator function which creates a new model (with number of input features as a parameter)
        :param opt_gen: a generator function which creates a new optimizer (with model as a parameter)
        """
        self.model_gen = model_gen
        self.opt_gen = opt_gen
        self.models = []

    def predict(self, X):
        """
        Predicts the class for each datapoint (row of X)
        :param X: input data
        :return:
        """
        # TODO
        return None

    def train(self, X, y):
        """
        Trains one-vs-all classifier (separate logistic regression for each class)
        :param X: input data
        :param y: gold classes
        :return:
        """
        # TODO
        pass

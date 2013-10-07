import numpy as np
import matplotlib.pyplot as pl

from numpy.random import multivariate_normal as mvg

class Collection:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

class Kernel:
    def __init__(self, name, parameters={}):
        def rbf(x, y):
            l = 1
            if 'l' in parameters:
                l = parameters['l']
            return exp(-1.0/(2*l*l)*np.linalg.norm(x - y)**2)
        def dot(x, y):
            return np.dot(x, y) + 1
        funcs = {
            'rbf' : rbf,
            'dot' : dot
        }
        self.__call__ = funcs[name]


class GP:
    def __init__(self, m, k, parameters={}):
        self.m = m
        self.k = k
        self.num_points = 0
        self.parameters = parameters

    def sample(self, points, num=1):
        n = len(points)
        if self.num_points > 0:
            kyy = self.gen_covmat(points, points)
            kyx = self.gen_covmat(points, self.x)
            kxy = self.gen_covmat(self.x, points)
            kxx = np.linalg.inv(self.kxx)
            covmat = kyy - kyx*kxx*kxy
            y = np.transpose([self.y])
            mean = kyx*kxx*y
            mean = np.array(np.transpose(mean))[0]
        else:
            covmat = self.gen_covmat(points, points)
            mean = np.zeros(n)
        return mvg(mean, covmat, num)

    def gen_covmat(self, m1, m2):
        return np.matrix([[self.k(x, y) for x in m2] for y in m1])

    def train(self, collection):
        self.x, self.y = collection.x, collection.y
        self.num_points = len(self.x)
        if 'noise' in self.parameters:
            noise = np.identity(len(self.x)) * self.parameters['noise']
        else:
            noise = np.zeros((len(self.x), len(self.x)))
        self.kxx = self.gen_covmat(self.x, self.x) + noise

if __name__ == "__main__":
    from math import exp
    def m(x):
        return 0
    gp = GP(m, Kernel('rbf', parameters={'l' : 2}), parameters={
        "noise" : 0.01,
    })
    train = Collection([1, 2, 3, 4],[1, 4, 9, 16])
    #train = Collection([0],[0])
    gp.train(train)
    x = np.arange(-5, 10, 0.05)
    points = np.array(x)
    samples = gp.sample(points, 100)
    [pl.plot(x, samples[i]) for i in range(len(samples))]
    pl.plot(train.x, train.y, 'ro')
    pl.show()

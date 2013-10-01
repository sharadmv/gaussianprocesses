import numpy as np
import matplotlib.pyplot as pl

from numpy.random import multivariate_normal as mvg

class GP:
    def __init__(self, m, k):
        self.m = m
        self.k = k

    def sample(self, points, num=1):
        n = len(points)
        covmat = np.zeros((n, n))
        covmat = np.array([[self.k(x, y) for x in points] for y in points])
        mean = np.zeros(n)
        return mvg(mean, covmat, num)

if __name__ == "__main__":
    from math import exp
    def k(x, y):
        return exp(-1/2 * np.linalg.norm(x - y)**2)
    def m(x):
        return 0
    gp = GP(m, k)
    points = np.array([[1, 1], [5, 6]])
    samples = gp.sample(points, 50)
    pl.plot(samples)
    pl.show()

import numpy as np

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
    def k(x, y):
        return np.dot(x, y)
    def m(x):
        return 0
    gp = GP(m, k)
    points = ((1, 2), (3, 4), (5, 6))
    x = gp.sample(points)

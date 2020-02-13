import numpy as np
import pandas as pd
import scipy.special
import sklearn.metrics
import math


def calc_grad(X_b, y_b, theta):
    z = np.inner(X_b,theta)
    h = scipy.special.expit(z)
    grd = np.inner(X_b.T, h-y_b)/y_b.shape[0]
    return grd


def predict(Xs, theta):
    z = np.inner(Xs,theta)
    h = scipy.special.expit(z)
    return h


def eval(X, y, theta):
    yh = predict(X, theta)
    score = sklearn.metrics.log_loss(y, yh)
    return score


class AdamTrainer:
    def __init__(
            self,
            lr=0.1,
            batch_size=512,
            eps=1e-8,
            b1=0.9,
            b2=0.999,
            seed=0
    ):
        self.eps = eps
        self.lr = lr
        self.batch_size = batch_size
        self.b1 = b1
        self.b2 = b2
        self.random = np.random.RandomState(seed=seed)

        self.t = 1
        self.b1t = b1
        self.b2t = b2
        self.mt = None
        self.vt = None

    def train_epoch(self, X, y, theta=None):
        r_order = self.random.permutation(len(X))
        X = X[r_order]
        y = y[r_order]
        n_rows = len(X)
        d = X.shape[1]

        batch_size = self.batch_size
        b1 = self.b1
        b2 = self.b2

        if theta is None:
            theta = np.zeros(shape=d)
        if self.mt is None:
            self.mt = np.zeros(shape=d)
        if self.vt is None:
            self.vt = np.zeros(shape=d)

        start_idx = 0
        for batch_idx in range(n_rows // batch_size):
            X_b = X[start_idx:start_idx + batch_size]
            y_b = y[start_idx:start_idx + batch_size]
            cur_grad = calc_grad(X_b, y_b, theta)
            self.mt *= b1
            self.mt += (1 - b1) * cur_grad
            #         mt = b1*mt + (1-b1)*cur_grad
            self.vt *= b2
            self.vt += (1 - b2) * (cur_grad * cur_grad)
            #         vt = b2*vt + (1-b2)*(cur_grad * cur_grad)
            at = (self.lr / math.sqrt(self.t)) * np.sqrt(1 - self.b2t) / (1 - self.b1t)
            #         at = lr * np.sqrt(1-b2t)/(1-b1t)
            theta -= at * self.mt / (np.sqrt(self.vt) + self.eps)

            start_idx += batch_size
            self.b1t *= b1
            self.b2t *= b2
            self.t += 1

        return theta

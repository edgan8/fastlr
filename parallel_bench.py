import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.metrics
import scipy.special
import numpy as np
import time
import math
from tqdm import tqdm as tqdm

import ray

def predict(Xs, theta):
    z = np.inner(Xs,theta)
    h = 1/(1+np.exp(-z))
    return h

def calc_grad(X, y, theta):
    z = np.dot(X,theta)
    h = 1/(1+np.exp(-z))
    grd = np.dot(X.T, h-y)/y.shape[0]
    return grd

class AdamOptimizer:
    def __init__(self, ndims, lr=.1):
        self.eps = 1e-8
        self.b1 = 0.9
        self.b2 = 0.999
        self.lr = lr
        self.ndims = ndims
        
        self.mt = np.zeros(shape=(ndims))
        self.vt = np.zeros(shape=(ndims))
        self.theta = np.zeros(shape=(ndims))
        self.t = 1
        self.b1t = 1
        self.b2t = 1

    def step(self, grad):
        b1 = self.b1
        b2 = self.b2
        
        self.mt *= b1
        self.mt += (1-b1)*grad
        self.vt *= b2
        self.vt += (1-b2)*(grad*grad)
        self.b1t *= b1
        self.b2t *= b2
        
        at = (self.lr/math.sqrt(self.t))*np.sqrt(1-self.b2t)/(1-self.b1t)
        self.theta -= at*self.mt/(np.sqrt(self.vt)+self.eps)
        self.t += 1
        return self.theta

    def get_params(self):
        return self.theta


@ray.remote
class AdamParameterServer(object):
    def __init__(self, dim, lr):
        self.opt = AdamOptimizer(dim, lr=lr)

    def get_params(self):
        return self.opt.get_params()

    def update_params(self, grad):
        self.opt.step(grad)


@ray.remote
def gradient_worker(ps, X, y, batch_size):
    n_batches = X.shape[0] // batch_size
    start_idx = 0
    
    for batch_idx in range(n_batches):
        X_b = X[start_idx:start_idx+batch_size]
        y_b = y[start_idx:start_idx+batch_size]
        cur_theta = ray.get(ps.get_params.remote())
        cur_grad = calc_grad(X_b, y_b, cur_theta)
        ps.update_params.remote(cur_grad)

        start_idx += batch_size

def train_remote(X, y, num_processes, batch_size):
    ray.init(num_cpus=5)
    X_parts = np.array_split(X, num_processes)
    y_parts = np.array_split(y, num_processes)
    X_ids = [ray.put(X_part) for X_part in X_parts]
    y_ids = [ray.put(y_part) for y_part in y_parts]
    
    ps = AdamParameterServer.remote(dim=X.shape[1], lr=.1)
    
    start_time = time.time()
    workers = [
        gradient_worker.remote(ps, X_ids[i], y_ids[i], batch_size)
        for i in range(num_processes)
    ]
    worker_res = ray.get(workers)
    end_time = time.time()
    print("Elapsed Time: {}".format(end_time - start_time))
    res = ray.get(ps.get_params.remote())
    ray.timeline("timeline.json")
    ray.shutdown()
    return res

def train_local(X, y, batch_size):
    n_rows = X.shape[0]
    ndims = X.shape[1]
    start_idx = 0
    lr = 0.1
    start_time = time.time()
    opt = AdamOptimizer(ndims, lr=lr)
    for batch_idx in tqdm(range(n_rows // batch_size)):
        X_b = X[start_idx:start_idx+batch_size]
        y_b = y[start_idx:start_idx+batch_size]
        cur_grad = calc_grad(X_b, y_b, opt.get_params())
        opt.step(cur_grad)
        
        start_idx += batch_size
    end_time = time.time()
    print("Total Time: {}".format(end_time - start_time))
    return opt.get_params()

def load_data():
    df = pd.read_feather("~/data/avazu/train_10M.feather")
    target = "click"
    CAT_COLS = [
        "C1", "banner_pos", 
        "site_category", "app_category", 
        "device_type", "device_conn_type",
    ]
    df_enc = pd.get_dummies(df[CAT_COLS], columns=CAT_COLS)
    df_final = pd.concat([
        df[target], df_enc
    ], axis=1)

    nrows = 4_000_000
    np.random.seed(0)
    r_order = np.random.permutation(nrows)
    Xs = df_enc.values[:nrows][r_order]
    y = df[target].values[:nrows][r_order]

    Xc = np.concatenate([
        np.repeat(1, repeats=Xs.shape[0]).reshape(-1,1),
        Xs
    ], axis=1)
    return Xc, y

def evaluate(X, y, theta):
    yh = predict(X, theta)
    score = sklearn.metrics.log_loss(y, yh)
    print("Log Loss: {}".format(score))

def main():
    X,y = load_data()
    batch_size = 10000
    # 1.7 seconds
    # theta = train_local(X, y, batch_size=10000)
    theta = train_remote(X, y, batch_size=10000, num_processes=4)
    evaluate(X, y, theta)

if __name__ == "__main__":
    main()
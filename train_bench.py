import pandas as pd
import numpy as np
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
torch.set_num_threads(1)

import sklearn.metrics
import scipy.special


def load_data():
    df = pd.read_csv(
        "~/data/avazu/train",
        nrows=1000000
    )
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
    Xs = df_enc.values
    y = df[target].values
    return Xs,y


class LinearModel(nn.Module):
    def __init__(self, k):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(k, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

def train_torch():
    Xs, y = load_data()
    print("Loaded Data")
    xt = torch.from_numpy(Xs.astype(np.float32))
    yt = torch.from_numpy(y)

    train_data = utils.TensorDataset(xt,yt)
    train_loader = utils.DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
        num_workers=1,
    )
    
    net = LinearModel(66)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    start_time = time.time()
    num_epochs = 1
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print("Running Loss: {}".format(running_loss))
    end_time = time.time()
    print('Finished Training in :{}'.format((end_time - start_time)))

def calc_grad(X_b, y_b, theta):
    z = np.inner(X_b,theta)
    h = scipy.special.expit(z)
    grd = np.inner(X_b.T, h-y_b)/y_b.shape[0]
    return grd

def predict(Xs, theta):
    z = np.inner(Xs,theta)
    h = scipy.special.expit(z)
    return h

def train_numpy():
    Xs, y = load_data()
    Xc = np.concatenate([
        np.repeat(1, repeats=Xs.shape[0]).reshape(-1,1),
        Xs
    ], axis=1)
    r_order = np.random.permutation(len(Xs))
    Xc = Xc[r_order]
    y = y[r_order]
    
    lr = 0.1
    batch_size = 512

    eps = 1e-8
    b1 = 0.9
    b2 = 0.999

    t1 = time.time()
    theta = np.zeros(shape=(Xc.shape[1]))
    mt = np.zeros(shape=(Xc.shape[1]))
    vt = np.zeros(shape=(Xc.shape[1]))
    b1t = b1
    b2t = b2

    n_rows = len(Xc)
    start_idx = 0
    t = 1
    for batch_idx in range(n_rows // batch_size):
        X_b = Xc[start_idx:start_idx+batch_size]
        y_b = y[start_idx:start_idx+batch_size]
        cur_grad = calc_grad(X_b, y_b, theta)
        mt = b1*mt + (1-b1)*cur_grad
        vt = b2*vt + (1-b2)*(cur_grad * cur_grad)
        at = (lr/math.sqrt(t))*np.sqrt(1-b2t)/(1-b1t)
#         at = lr * np.sqrt(1-b2t)/(1-b1t)
        theta -= at*mt/(np.sqrt(vt)+eps)

        start_idx += batch_size
        b1t *= b1
        b2t *= b2
        t += 1
    
    t2 = time.time()
    print("Total Time: {}".format(t2-t1))
    yh = predict(Xc, theta)
    score = sklearn.metrics.log_loss(y, yh)
    print("Log Loss: {}".format(score))
    
def main():
#     train_torch()
    train_numpy()

if __name__ == "__main__":
    main()
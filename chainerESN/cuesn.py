import numpy as np
import cupy as cp
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, Chain


class ESN(Chain):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_reservoir=200,
        leaking_rate=0.3,
    ):
        super(ESN, self).__init__()
        
        # Hyperparams
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.leaking_rate = leaking_rate
        
        # activation function for resv
        self.resv_act = F.tanh
        self.resv_stacked = xp.array()
        self.y_hat_stacked = xp.array()
        
        # params
        self.Wi = np.array(
            self.randst.rand(n_inputs, n_reservoir) - 0.5,
            dtype=xp.float32
        )
        
        self.Wr = np.array(
            self.randst.rand(n_reservoir, n_reservoir) - 0.5,
            dtype=xp.float32
        )
        
        self.Wo = np.array(
            self.randst.rand(n_reservoir, n_outputs) - 0.5,
            dtype=xp.float32
        )
        
    def __call__(self, data_x, prev_resv, data_y):
        # compute forward
        y_hat, resv = self.forward(data_x, prev_resv)
        self.resv_stacked.append(resv)
        self.y_hat_stacked.append(y_hat)
        return y_hat, resv
    
    def forward(self, data_x, prev_resv):
        u = F.matmul(data_x, self.Wi)
        pre_act_resv = F.matmul(prev_resv, self.Wr) + u
        resv = (1 - self.leaking_rate) * prev_resv + self.leaking_rate * self.resv_act(pre_act_resv)
        prev_resv = resv
        return y, resv
    
def fit(model, data_x):
    resv = xp.zeros()
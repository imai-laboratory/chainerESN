import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class ESN(Chain):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_reservoir=200,
        leaking_rate=0.3,
        out_activation='affine'
    ):
        super(ESN, self).__init__()
        
        # Hyperparams
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.leaking_rate = leaking_rate
        
        # activation function for resvoir
        self.resv_act = F.tanh
       
        # todo: implement random component selector
        self.randst = np.random.RandomState(1234)
        self.Wr = Variable(
            self.randst.rand(n_reservoir, n_reservoir) - 0.5,
            dtype=float32
        )
        
        with self.init_scope():
            self.l1 = L.Linear(n_inputs, n_reservoir)
            self.l3 = L.Linear(n_reservoir, n_outputs)

    def __call__(self, data_x, prev_resv, data_y):
        u = self.l1(data_x)  # n_input -> n_resv
        pre_act_resv = (1 - self.leaking_rate) * (prev_resv.dot(self.Wr)) + leaking_rate * u
        resv = self.resv_act(pre_act_resv)
        prev_resv = resv
        y = self.l3(resv)
        
        # define loss
        loss = F.softmax_cross_entropy(y, data_y)
        accuracy = F.accuracy(y, data_y)
        
        # report loss
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss
        
        
if __name__ == '__main__':
    esn = ESN(
        n_inputs=100,
        n_outputs=10,
        n_reservoir=200,
        leaking_rate=0.3,
    )
    optimizer = optimizers.SGD()
    optimizer.setup(model)
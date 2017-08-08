import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, Chain
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
        self.Wi = Variable(
            np.array(self.randst.rand(n_inputs, n_reservoir) - 0.5,
                     dtype=np.float32
                    )
        )
        self.Wr = Variable(
            np.array(self.randst.rand(n_reservoir, n_reservoir) - 0.5,
                     dtype=np.float32
                    )
        )
        
        with self.init_scope():
            # self.l1 = L.Linear(n_inputs, n_reservoir)
            self.l3 = L.Linear(n_reservoir, n_outputs)

    def __call__(self, data_x, prev_resv, data_y):
        # compute forward
        output_y, resv = self.forward(data_x, prev_resv)
        
        # compute loss
        loss = F.mean_squared_error(output_y, data_y)
        # accuracy = F.accuracy(output_y, data_y)
        
        # report loss
        # report({'loss': loss, 'accuracy': accuracy}, self)
        report({'loss': loss}, self)
        return output_y, loss, resv
        
    def forward(self, data_x, prev_resv):
        # u = self.l1(data_x)  # n_input -> n_res
        # print(data_x)
        # print('type {} {}'.format(prev_resv.shape, data_x.data.shape))
        with chainer.using_config('enable_backprop', False):
            u = F.matmul(data_x, self.Wi)
            pre_act_resv = F.matmul(prev_resv, self.Wr) + u
            # data_x (batch, n_input)
            # Wi (n_input, n_hidden)
            # u = data_x * self.Wi
            # pre_act_res = prev_resv * self.Wr + u
            resv = (1 - self.leaking_rate) * prev_resv + self.leaking_rate * self.resv_act(pre_act_resv)
            prev_resv = resv
        y = F.sigmoid(self.l3(resv))
        return y, resv
    
    
def fit(model, data_x, optimizer, n_train_epochs):
    resv = Variable(np.zeros((1, model.n_reservoir), dtype=np.float32))
    data_x = np.array(data_x, dtype=np.float32)
    # print('fit shape {}'.format(data_x[:, 1].shape))
    for e in range(n_train_epochs):
        model.zerograds()  # initialize gradients
        _, loss, resv = model(data_x[:, e:e+1], resv, data_x[:, e:e+1])  # __call__ function forward computation
        print('loss: {}'.format(loss.data))
        loss.backward()  # backward computation, computing gradients
        optimizer.update()  # update params
        
def generate(model, data_x, n_init_frames):
    # (1, 2000)
    data_x = np.array(data_x, dtype=np.float32)
    init_data, test_data = data_x[:, :n_init_frames], data_x[:, n_init_frames:]
    print('shape: {}, {}'.format(init_data.shape, test_data.shape))
    x_size = test_data.shape[1]
    resv = Variable(np.zeros((1, model.n_reservoir), dtype=np.float32))
    y_hat_chain = []
    mse_chain = []
    
    
    # initialize resv
    print('initializing {} frames'.format(n_init_frames))
    for t in range(n_init_frames):
        _y_hat, resv = model.forward(init_data[:, t:t+1], resv)
        
    # prediction
    print('making predctions for {} iterations'.format(x_size))
    for t in range(x_size):
        y_hat, loss, resv = model(test_data[:, t:t+1], resv, test_data[:, t:t+1])
        y_hat_chain += [y_hat]
        mse_chain += [loss.data]
        print(loss.data)
        
    return mse_chain
    
if __name__ == '__main__':
    dataset = np.loadtxt('MackeyGlass_t17.txt')
    train_data, test_data = np.split(dataset, [8000])
    train_data = np.reshape(train_data, (1, train_data.shape[0]))
    test_data = np.reshape(test_data, (1, test_data.shape[0]))
    
    esn = ESN(
        n_inputs=1,
        n_outputs=1,
        n_reservoir=200,
        leaking_rate=0.3,
    )
    
    optimizer = optimizers.SGD()
    optimizer.setup(esn)
    fit(esn, train_data, optimizer, 100)
    generate(esn, test_data, 100)
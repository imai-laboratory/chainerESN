import numpy as np
import chainer
import chainer.functions as F
from chainer import Variable, Chain
import scipy.linalg
import matplotlib.pyplot as plt

GPUFLAGS = False

if GPUFLAGS:
    import cupy as cp
    xp = cp
else:
    xp = np


class ESN(Chain):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_reservoir=200,
        leaking_rate=0.3,
        randst=np.random
    ):
        super(ESN, self).__init__()
        # Hyperparams
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.leaking_rate = leaking_rate

        # activation function for resv
        self.resv_act = F.tanh

        self.randst = randst
        self.randst.seed(101)

        # params
        self.Wi = np.array(
            self.randst.rand(n_reservoir, n_inputs + 1) - 0.5,
            dtype=xp.float32
        )

        self.Wr = np.array(
            self.randst.rand(n_reservoir, n_reservoir) - 0.5,
            dtype=xp.float32
        )

        self.Wr *= 0.135  # MAGIC!!

        self.Wo = None

        self.resv = np.zeros((self.n_reservoir, 1), dtype=xp.float32)

    def __call__(self, data_x):
        '''
        the function to compute forward inference
        '''
        r = self.update(data_x)
        y = xp.dot(self.Wo, F.vstack(
            (xp.array([[1]], dtype=xp.float32), data_x, r)))
        return y

    def update(self, data_x):
        u = F.matmul(self.Wi, F.vstack(
            ((xp.array([[1]], dtype=xp.float32), data_x))))
        r_tld = self.resv_act(F.matmul(self.Wr, self.resv) + u)
        new_r = (1 - self.leaking_rate) * self.resv \
            + self.leaking_rate * r_tld
        self.resv = new_r
        return new_r

    def clear_resv(self):
        self.resv = np.zeros((self.n_reservoir, 1))


def fit(model, data_x, data_y, init_len):
    '''
    data_x: input with the shape of (samples, channel)
    data_y: teacher signal with the shape of (samples, channel)
    '''
    train_len = len(data_x)
    X = xp.zeros(
        (1 + model.n_inputs + model.n_reservoir, train_len - init_len),
        dtype=xp.float32
    )
    Yt = data_y[init_len: train_len + 1]

    # forward computation
    for i in range(train_len):
        u = data_x[i]
        u = F.reshape(u, (-1, 1))
        r = model.update(u)
        if i >= init_len:
            # collect data after initialization
            '''
            s = F.vstack(
                (xp.array([[0]], dtype=xp.float32), u, r
                 ))[:, 0].data
            '''
            X[:, i-init_len] = F.vstack(
                (xp.array([[0]], dtype=xp.float32), u, r)
            )[:, 0].data
    # linear regression: X and Yt
    model.Wo = xp.dot(Yt[:, 0], scipy.linalg.pinv(X))


def forward(
        model,
        test_data_x,
        init_len,
        test_len,
        generative=False
):
    '''
    the function to compute forward ihference
    '''
    assert init_len + test_len < len(test_data_x)
    if 0 < init_len:
        model.clear_resv()

    Y = xp.zeros((model.n_outputs, test_len))

    u = test_data_x[0]
    for i in range(init_len + test_len - 1):
        u = F.reshape(u, (-1, 1))
        y = model(u)
        if i >= init_len:
            Y[:, i] = y[0].data

        if generative:
            u = y[0].data
        else:
            u = test_data_x[i + 1]

    return Y


if __name__ == '__main__':
    chesn = ESN(
        n_inputs=1,
        n_outputs=1,
        n_reservoir=200,
        leaking_rate=0.3
    )
    data = np.loadtxt('MackeyGlass_t17.txt').reshape((-1, 1))
    data = data.astype(xp.float32)
    train_set = data[:8000]
    test_set = data[8000:]
    fit(chesn, train_set, train_set, 300)
    Y = forward(chesn, test_set, 0, 500, generative=True)

    # plot some signals
    plt.figure(1).clear()
    plt.plot(test_set[:500], 'g')
    plt.plot(Y.T, 'b')
    plt.title('Target and generated signals $y(n)$ starting at $n=0$')
    plt.legend(['Target signal', 'Free-running predicted signal'])

    '''
    plt.figure(2).clear()
    plt.plot(X[0:20, 0:200].T)
    plt.title('Some reservoir activations $\mathbf{x}(n)$')

    plt.figure(3).clear()
    plt.bar(range(1 + inSize + resSize), Wout.T)
    plt.title('Output weights $\mathbf{W}^{out}$')
    '''
    plt.show()

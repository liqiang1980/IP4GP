# filter
from scipy import zeros, signal
import matplotlib.pyplot as plt
import numpy as np


class lfilter:

    def __init__(self, order, fs, init_value, n_dim):
        self.b = []
        self.z = []
        for i in range(n_dim):
            self.b.append(signal.firwin(order + 1, fs))
            self.z.append(list(signal.lfilter_zi(self.b[i], 1) * init_value[i]))

    def lp_filter(self, x, n_dim):
        res2 = np.zeros(n_dim)
        z = []
        for i in range(n_dim):
            result, self.z[i] = signal.lfilter(self.b[i], 1, [x[i]], zi=self.z[i])
            res2[i] = result[0]
            z.append(self.z[i])
        return res2, z

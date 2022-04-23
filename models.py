from modules import *

class ARM_CML():
    def __init__(self, dim, n_in, n_out, n_hid, kernel_size):
        if dim == 2:
            self.f_cont = ContextNet2D(n_in, n_out, n_hid, kernel_size)
            self.f_pred = CNN(n_in, n_out, n_hid, kernel_size)
import os
import torch

os.environ["KERAS_BACKEND"] = "torch"

import keras

from keras import backend
from keras import ops
from keras.src.backend.torch.core import *

import scipy.sparse

class LayerELSA(keras.layers.Layer):
    def __init__(self, n_dims, n_items, device):
        super(LayerELSA, self).__init__()
        self.device = device
        self.A = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty([n_dims, n_items])))
        self.W_list=[self.A]
    
    def parameters(self, recurse=True):
        return self.W_list
        
    def get_weights_(self):
        return keras.ops.vstack([self.W_list])
        
    def track_module_parameters(self):
        for param in self.parameters():
            variable = keras.Variable( #keras.backend.Variable(
                initializer=param, trainable=param.requires_grad
            )
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def build(self):
        self.to(self.device)
        sample_input = torch.ones([self.A.shape[0]]).to(self.device)
        _ = self.call(sample_input)
        self.track_module_parameters()

    def call(self, x):
        A = self.get_weights()
        A = torch.nn.functional.normalize(A, dim=-1)
        xA = torch.matmul(x, A)
        xAAT = torch.matmul(xA, A.T)
        return keras.activations.relu(xAAT - x)

class LayerANNA(keras.layers.Layer):
    def __init__(self, n_dims, n_items, device):
        super().__init__()
        self.device = device
        self.A = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty([n_dims, n_items])))
        self.B = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty([n_dims, n_items])))

    def parameters(self, recurse=True):
        return [self.A, self.B]

    def track_module_parameters(self):
        for param in self.parameters():
            variable = keras.Variable( #keras.backend.Variable(
                initializer=param, trainable=param.requires_grad
            )
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def build(self):
        self.to(self.device)
        sample_input = torch.ones([self.A.shape[0]]).to(self.device)
        _ = self.call(sample_input)
        self.track_module_parameters()

    def call(self, x):
        A = self.A
        B = self.B
        xA = torch.matmul(x, A)
        xABT = torch.matmul(xA, B.T)
        return keras.activations.relu(xABT - x*torch.sum(A*B, dim=-1)) #mozna?
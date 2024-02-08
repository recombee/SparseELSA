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

    def parameters(self, recurse=True):
        return [self.A]

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
        A = torch.nn.functional.normalize(self.A, dim=-1)
        xA = torch.matmul(x, A)
        xAAT = torch.matmul(xA, A.T)
        return keras.activations.relu(xAAT - x)

class SparseLayerELSA(keras.layers.Layer):
    def __init__(self, n_items, n_dims, device, embeddings=None):
        super(SparseLayerELSA, self).__init__()
        self.device = device
        if embeddings is not None:
            print("create layer from provided embeddings")
            assert embeddings.shape[0] == n_items
            assert embeddings.shape[1] == n_dims
            self.A = torch.nn.Parameter(
                torch.from_numpy(embeddings)
            )
        else:
            print("create new layer from scratch")
            self.A = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.empty([n_items, n_dims])))
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
        A = self.A
        A = torch.nn.functional.normalize(A, dim=-1)
        xA = torch.matmul(x, A)
        xAAT = torch.matmul(xA, A.T)
        return keras.activations.relu(xAAT - x)


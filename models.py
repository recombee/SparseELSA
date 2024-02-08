import os
import torch

os.environ["KERAS_BACKEND"] = "torch"

import keras

from keras import backend
from keras import ops
from keras.src.backend.torch.core import *

import scipy.sparse

from layers import LayerELSA, SparseLayerELSA 
from datasets.pydatasets import BasicRecSysDataset, PredictDfRecSysDataset, SparseRecSysDataset, SparseTransposedRecSysDataset
from datasets.utils import *

def NMSE(x,y):
    x=torch.nn.functional.normalize(x, dim=-1)
    y=torch.nn.functional.normalize(y, dim=-1)
    return keras.losses.mean_squared_error(x,y)

class KerasELSA(keras.models.Model):
    def __init__(self, n_items, n_dims, items_idx, device):
        super().__init__()
        #self.A = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty([n_items, n_dims])))
        self.device = device
        self.ELSA = LayerELSA(n_items, n_dims, device=device)
        self.items_idx = items_idx
        self.ELSA.build()

    def call(self, x):
        return self.ELSA(x)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()

        # Compute loss
        y_pred = self(x, training=True)  # Forward pass
        loss = self.compute_loss(y=y, y_pred=y_pred)

        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        #print({m.name: m.result() for m in self.metrics})
        return {m.name: m.result() for m in self.metrics}

    def predict_sparse(self, x):
        data = BasicRecSysDataset(x)
        return self.predict(data)

    def predict_df(self, df, k=100, user_ids=None):

        if user_ids is None:
            user_ids = np.array(df.user_id.cat.categories)

        #x = get_sparse_matrix_from_dataframe(df, item_indices=self.items_idx)

        data = PredictDfRecSysDataset(df, self.items_idx)

        dfs = []
        imin = 0
        auser_ids = user_ids

        for i in tqdm(range(len(data)), total=len(data)):
            x, batch_uids = data[i]

            batch = torch.from_numpy(self.predict_on_batch(x))
            mask = 1-x.astype(bool)  # block reminder
            batch = batch * mask

            values_, indices_ = torch.topk(batch.to("cpu"), k)
            df = pd.DataFrame({"user_id": np.stack([batch_uids]*k).flatten("F"), "item_id": np.array(
                self.items_idx)[indices_].flatten(), "value": values_.flatten()})
            df["user_id"] = df["user_id"].astype(str).astype('category')
            df["item_id"] = df["item_id"].astype(str).astype('category')
            dfs.append(df)

        df = pd.concat(dfs)
        df["user_id"] = df["user_id"].astype(str).astype('category')
        df["item_id"] = df["item_id"].astype(str).astype('category')
        return df

class SparseKerasELSA(keras.models.Model):
    """
    Same as KerasELSA but receives data from SparseRecSysDataset - data is batch of user vectors + slicer for nonzero entries
    """
    def __init__(self, n_items, n_dims, items_idx, device, top_k=1500):
        super().__init__()
        #self.A = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty([n_items, n_dims])))
        self.device = device
        self.ELSA = SparseLayerELSA(n_items, n_dims, device=device)
        self.items_idx = items_idx
        self.ELSA.build()
        self(np.zeros([1,n_items]))
        self.finetuning = False
        self.top_k = top_k

    def call(self, x):
        return self.ELSA(x)

    def forward_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        
        if len(data)==2:
            full_x = None
            a,b = data
            x, y = a
            y = torch.hstack((x,y))
            slicer, negative_slicer = b
                
        elif len(data)==3:
            full_x, slicer, negative_slicer = data
        else:
            full_x, slicer = data
            negative_slicer = None
            
        
        #full_x=full_x.to(self.device)
        if full_x is not None:
            if negative_slicer is not None:
                y = full_x[:, negative_slicer]
            else:
                y = full_x

            x = full_x[:, slicer]


            x = x.to(self.device)
            y = y.to(self.device)
        
        if negative_slicer is not None:
            negative_slicer = negative_slicer.to(self.device)
        
        slicer=slicer.to(self.device)

        return x, y, slicer, negative_slicer
        
    def train_step(self, data):
        
        #x, y, slicer, negative_slicer = self.forward_step(data)
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        
        if len(data)==2:
            full_x = None
            a,b = data
            x, y = a
            y = torch.hstack((x,y))
            slicer, negative_slicer = b
                
        elif len(data)==3:
            full_x, slicer, negative_slicer = data
        else:
            full_x, slicer = data
            negative_slicer = None
            
        
        #full_x=full_x.to(self.device)
        if full_x is not None:
            if negative_slicer is not None:
                y = full_x[:, negative_slicer]
            else:
                y = full_x

            x = full_x[:, slicer]


            x = x.to(self.device)
            y = y.to(self.device)
        x_out=y
        
        #print(x.shape, y.shape, slicer.shape)
        #print(x.shape)
        #print(full_x.shape)
        #print(slicer)
        
        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()

        # Compute loss
        #y_pred = self(x, training=True)  # Forward pass
        
        A = self.ELSA.A
        #print(A.shape)
        A_slicer = A[slicer]
        A_slicer = torch.nn.functional.normalize(A_slicer, dim=-1)
        
        if negative_slicer is not None:
            A_negative_slicer = A[negative_slicer]
            A_negative_slicer = torch.nn.functional.normalize(A_negative_slicer, dim=-1)
        else:
            A_negative_slicer = torch.nn.functional.normalize(A, dim=-1)
            
        xA = torch.matmul(x, A_slicer)
        #print(xA.shape)
        
        xAAT = torch.matmul(xA, A_negative_slicer.T)
        
        y_pred = keras.activations.relu(xAAT - x_out, max_value=6)

        if self.finetuning:
            val, inds = torch.topk(y_pred, self.top_k)
            y = torch.gather(y,1,inds)
            y_pred = val
            
        
        loss = self.compute_loss(y=y, y_pred=y_pred)
        
        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        #print({m.name: m.result() for m in self.metrics})
        return {m.name: m.result() for m in self.metrics}

    def predict_sparse(self, x):
        data = BasicRecSysDataset(x)
        return self.predict(data)

    def predict_df(self, df, k=100, user_ids=None):

        if user_ids is None:
            user_ids = np.array(df.user_id.cat.categories)

        #x = get_sparse_matrix_from_dataframe(df, item_indices=self.items_idx)

        data = PredictDfRecSysDataset(df, self.items_idx, batch_size=1024)

        dfs = []
        imin = 0
        auser_ids = user_ids

        for i in tqdm(range(len(data)), total=len(data)):
            x, batch_uids = data[i]

            batch = torch.from_numpy(self.predict_on_batch(x))
            mask = 1-x.astype(bool)  # block reminder
            batch = batch * mask

            values_, indices_ = torch.topk(batch.to("cpu"), k)
            df = pd.DataFrame({"user_id": np.stack([batch_uids]*k).flatten("F"), "item_id": np.array(
                self.items_idx)[indices_].flatten(), "value": values_.flatten()})
            df["user_id"] = df["user_id"].astype(str).astype('category')
            df["item_id"] = df["item_id"].astype(str).astype('category')
            dfs.append(df)

        df = pd.concat(dfs)
        df["user_id"] = df["user_id"].astype(str).astype('category')
        df["item_id"] = df["item_id"].astype(str).astype('category')
        return df


import os

# for ALS MF
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"

from math import ceil
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets.utils import *
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
import implicit
import torch

class TopPopularRecommender:
    def __init__(self, X, item_idx):
        self.item_idx = item_idx
        self.popularities = np.array(X.sum(0))[0]
    
    def fit(*args, **kwargs):
        self.__init__(*args, **kwargs)
    
    def predict_df(self, df, n=100, batch_size=1000):
        X_test = get_sparse_matrix_from_dataframe(df, item_indices=self.item_idx)
        n_batches = ceil(X_test.shape[0]/batch_size)
        uids = df.user_id.cat.categories.to_numpy()
        dfs=[]
        for i in tqdm(range(n_batches)):
            i_min = i*batch_size
            i_max = i_min+batch_size
            batch=X_test[i_min:i_max]
            indices = np.argsort((1-batch.toarray())*self.popularities)[:,-n:]
            indices = indices[:,::-1]
            item_ids = self.item_idx.to_numpy()[indices]
            batch_user_ids = uids[i_min:i_max]
            values = self.popularities[indices]
            df=pd.DataFrame({
                "user_id": batch_user_ids,
                "item_id": list(item_ids),
                "value": list(values)
            })
            df = df.explode(["item_id", "value"])
            df["item_id"] = df["item_id"].astype(str).astype("category")
            df["user_id"] = df["user_id"].astype(str).astype("category")
            dfs.append(df)
            
        return pd.concat(dfs)

class KNNRecommender:
    def __init__(self, X, item_idx, neighbors):
        self.item_idx = item_idx
        self.X = X
        self.model = NearestNeighbors(algorithm="brute", n_neighbors=neighbors, metric="cosine")
        self.model.fit(X)
        self.neighbors =neighbors
    
    def predict(self, X, **kwargs):
        if X.count_nonzero() == 0:
            return np.random.uniform(size=X.shape)

        if kwargs.get("neighbors"):
            n_distances, n_indices = self.model.kneighbors(X, n_neighbors=kwargs.get("neighbors"))
        else:
            n_distances, n_indices = self.model.kneighbors(X)

        n_distances = 1 - n_distances

        sums = n_distances.sum(axis=1)
        n_distances = n_distances / sums[:, np.newaxis]

        def f(dist, idx):
            A = self.X[idx]
            D = sp.diags(dist)
            return D.dot(A).sum(axis=0)

        vf = np.vectorize(f, signature="(n),(n)->(m)")
        X_predict = vf(n_distances, n_indices)

        X_predict[X.nonzero()] = 0
        X_predict = np.array(X_predict)
        #filt = (1-X.toarray())
    
        return X_predict#*filt
    
    def predict_df(self, df, n=100, batch_size=1000, neighbors=None):
        if neighbors is None:
            neighbors = self.neighbors
        X_test = get_sparse_matrix_from_dataframe(df, item_indices=self.item_idx)
        n_batches = ceil(X_test.shape[0]/batch_size)
        uids = df.user_id.cat.categories.to_numpy()
        dfs=[]
        for i in tqdm(range(n_batches)):
            i_min = i*batch_size
            i_max = i_min+batch_size
            batch=X_test[i_min:i_max]
            preds=self.predict(batch, neighbors=neighbors)
            preds=preds*(1-batch.toarray())
            preds=torch.from_numpy(preds)
            batch_uids = uids[i_min:i_max]
            values_, indices_ = torch.topk(preds.to("cpu"), n)
            df = pd.DataFrame({"user_id": np.stack([batch_uids]*n).flatten("F"), "item_id": np.array(
                self.item_idx)[indices_].flatten(), "value": values_.flatten()})
            df["user_id"] = df["user_id"].astype(str).astype('category')
            df["item_id"] = df["item_id"].astype(str).astype('category')
            dfs.append(df)  
            """
            #indices = np.argsort((1-batch.toarray())*preds)[:,-n:]
            indices = np.argsort(preds)[:,-n:]
            indices = indices[:,::-1]
            item_ids = self.item_idx.to_numpy()[indices]
            batch_user_ids = uids[i_min:i_max]
            bb = []
            for i in range(preds.shape[0]):
                bb.append(preds[i][indices[i]])
            values =np.vstack(bb)
            df=pd.DataFrame({
                "user_id": batch_user_ids,
                "item_id": list(item_ids),
                "value": list(values)
            })
            df = df.explode(["item_id", "value"])
            df["item_id"] = df["item_id"].astype(str).astype("category")
            df["user_id"] = df["user_id"].astype(str).astype("category")
            dfs.append(df)
            """
        return pd.concat(dfs)

class ALSMatrixFactorizer():
    def __init__(self, factors: int, regularization: float, iterations: int, use_gpu: bool, item_idx, num_threads=0):
        self.model = None
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.use_gpu = use_gpu
        self.item_idx = item_idx
        self.num_threads = num_threads

    def name(self):
        return "MF"

    def fit(self, X, training=False):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
            num_threads=self.num_threads,

        )
        self.model.fit(X)

    def predict(self, X, **kwargs):
        if not isinstance(X, scipy.sparse.csr_matrix):
            X = scipy.sparse.csr_matrix(X)

        if X.count_nonzero() == 0:
            return np.random.uniform(size=X.shape)

        recommended_item_ids, scores = self.model.recommend(
            np.arange(X.shape[0]), X,
            recalculate_user=True, filter_already_liked_items=False,
            N=X.shape[1]
        )
        predicted_scores = np.zeros((X.shape[0], X.shape[1]))
        np.put_along_axis(predicted_scores, recommended_item_ids, scores, axis=1)
        min_score = scores.min()
        if min_score < 0:
            scores += abs(min_score) + 0.1
        predicted_scores[X.nonzero()] = 0 # Just to be sure
        return predicted_scores

    def predict_df(self, df, k=100, batch_size=1000):
        X_test = get_sparse_matrix_from_dataframe(df, item_indices=self.item_idx)
        n_batches = ceil(X_test.shape[0]/batch_size)
        uids = df.user_id.cat.categories.to_numpy()
        dfs=[]
        for i in tqdm(range(n_batches)):
            i_min = i*batch_size
            i_max = i_min+batch_size
            batch=X_test[i_min:i_max]
            preds=self.predict(batch)
            preds=preds*(1-batch.toarray())
            preds=torch.from_numpy(preds)
            batch_uids = uids[i_min:i_max]
            values_, indices_ = torch.topk(preds.to("cpu"), k)
            df = pd.DataFrame({"user_id": np.stack([batch_uids]*k).flatten("F"), "item_id": np.array(
                self.item_idx)[indices_].flatten(), "value": values_.flatten()})
            df["user_id"] = df["user_id"].astype(str).astype('category')
            df["item_id"] = df["item_id"].astype(str).astype('category')
            dfs.append(df)            
        return pd.concat(dfs)

class EASERecommender:
    def __init__(self, item_idx, lambda_):
        self.item_idx = item_idx
        self.lambda_ = lambda_

    def fit(self, X):
        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += self.lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0
        self.B = B

    def predict(self, X):
        return np.dot(X, self.B)

    def predict_df(self, df, k=100, batch_size=1000):
        X_test = get_sparse_matrix_from_dataframe(df, item_indices=self.item_idx)
        n_batches = ceil(X_test.shape[0]/batch_size)
        uids = df.user_id.cat.categories.to_numpy()
        dfs=[]
        for i in tqdm(range(n_batches)):
            i_min = i*batch_size
            i_max = i_min+batch_size
            batch=X_test[i_min:i_max].toarray()
            preds=self.predict(batch)
            preds=preds*(1-batch)
            preds=torch.from_numpy(preds)
            batch_uids = uids[i_min:i_max]
            values_, indices_ = torch.topk(preds.to("cpu"), k)
            df = pd.DataFrame({"user_id": np.stack([batch_uids]*k).flatten("F"), "item_id": np.array(
                self.item_idx)[indices_].flatten(), "value": values_.flatten()})
            df["user_id"] = df["user_id"].astype(str).astype('category')
            df["item_id"] = df["item_id"].astype(str).astype('category')
            dfs.append(df)            
        return pd.concat(dfs)
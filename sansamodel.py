import os
import sys
from math import ceil
sys.path.append(os.path.join(sys.path[0],"sansa"))
from sansa.models.sansa import SANSA
import torch
import pandas as pd
from datasets.utils import *
import math, typing

class SansaRecommender:
    def __init__(self, item_idx, lambda_, 
                 target_density=0, umr_scans=3, umr_finetune_steps=10, 
                 umr_loss_threshold=1e-4, ldlt_method="icf"):
        self.item_idx = item_idx
        self.lambda_ = lambda_

        self.model_config = {
            "l2": lambda_,  # L2 regularization
            "target_density": target_density,  # num_nonzeros / (num_items)^2 = 0.005%
            "ainv_params": {
                "umr_scans": umr_scans,  # number of refinement passes through entire matrix. later scans are more precise and expensive. 0-3 recommended
                "umr_finetune_steps": umr_finetune_steps,  # number of refinements improving "several worst columns". relatively inexpensive. typically 0-20
                "umr_loss_threshold": umr_loss_threshold,  # if loss reaches below threshold, training is finished
            },
            "ldlt_method": ldlt_method,  # also available "cholmod" -- more accurate & more expensive (cholmod is exact factorization + sparsification, icf is incomplete factorization)
        }

    def fit(self, X):
        if self.model_config["target_density"]==0:
            self.model_config["target_density"] = X.nnz/(X.shape[1]**2)
        self.model = SANSA.from_config(self.model_config)
        self.model.weights, construct_weights_time = self.model._construct_weights(X.T)

    def predict_df(self, df, k=100, batch_size=1000):
        item_idx = self.item_idx
        X_test = get_sparse_matrix_from_dataframe(df, item_indices=item_idx)
        n_batches = ceil(X_test.shape[0]/batch_size)
        uids = df.user_id.cat.categories.to_numpy()
        dfs=[]
        for i in tqdm(range(n_batches)):
            i_min = i*batch_size
            i_max = i_min+batch_size
            batch=X_test[i_min:i_max]
            preds=self.model._predict(batch)[0].toarray()
            preds[preds == -np.inf] = 0
            preds=preds*(1-batch.toarray())
            preds=torch.from_numpy(preds)
            batch_uids = uids[i_min:i_max]
            values_, indices_ = torch.topk(preds.to("cpu"), k)
            df = pd.DataFrame({"user_id": np.stack([batch_uids]*k).flatten("F"), "item_id": np.array(
                item_idx)[indices_].flatten(), "value": values_.flatten()})
            df["user_id"] = df["user_id"].astype(str).astype('category')
            df["item_id"] = df["item_id"].astype(str).astype('category')
            dfs.append(df)            
        return pd.concat(dfs)
    
    def similar_items(
        self,
        N: int,
        batch_size: int,
        sources: typing.Union[np.ndarray] = None,
        candidates: typing.Union[np.ndarray] = None,
        verbose: bool = True,
    ) -> tuple:
        """
        Calculate a list of similar items measured by a cosine similarity of item embeddings
        Parameters
        ----------
        N : int
            The number of similar items to return
        batch_size : int
            Number of source items computed in one batch
        sources : np.ndarray or torch.Tensor, optional
            One dimension array of item indices to select for which items the similar items should be computed
        candidates : np.ndarray or torch.Tensor, optional
            One dimension array of item indices to select which items that can be returned as one of the the most similar items
        Returns
        -------
        tuple
            Tuple of (itemids, scores) torch tensors allocated on cpu
                The dimensions both tensors are (num_items, N) where num_items is a number of items recognized by the model
                or (len(sources), N) when parameter 'sources' is passed
        """
        n_items=len(self.item_idx)
        if sources is None:
            sources = np.arange(n_items)

        if candidates is not None:
            candidates_vec = np.zeros(n_items, dtype="float32")
            candidates_vec[candidates] = 1.0
        else:
            candidates_vec = np.ones(n_items, dtype="float32")


        c_v = torch.from_numpy(candidates_vec)
        indices = []
        scores = []
        i = 0
        max_i = math.ceil(len(sources)/ batch_size)
        if verbose:
            #self.__logger.info(f"Number of batches with size {batch_size} to compute cosine similarity and predict TopK is {max_i}")
            print(f"Number of batches with size {batch_size} to compute cosine similarity and predict TopK is {max_i}")

        #for res in self.predict_generator(data=source_data, batch_size=batch_size):
        W, Z = self.model.weights
    
        for i in tqdm(range(max_i)):

            ind = i*batch_size
            
            ind_min = ind
            ind_max = ind+batch_size
            
            #if verbose and ((i + 1) % 100 == 0):
            #    #self.__logger.info(f"Batch {i + 1}/{max_i}, number of source items processed: {(i+1)*batch_size}")
            #    print(f"Number of batches with size {batch_size} to compute cosine similarity and predict TopK is {max_i}")
            
            # get predictions
            res = torch.from_numpy((W[sources[ind_min:ind_max], :]@Z).toarray())
            y_c = res * c_v
            vals, inds = torch.topk(y_c, N)
            indices.append(inds)
            scores.append(vals)
            i += 1

        scores = torch.vstack(scores)
        indices = torch.vstack(indices)
        return (indices.cpu(), scores.cpu())

        
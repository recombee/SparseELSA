# Recommender System Datasets (and Baselines)

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

1. In the datasets/amazon directory run `download_amazon`, unpack `dups.zip` and run `preprocess.py`, `merge_ratings.py` and `merge_items.py`.
2. Use `experiment_*.py` to run experiments, for example `python experiment_knn.py --dataset amazon` will run knn on amazon dataset.
3. You can also setup experiments in json config file and run it in batches: `python run.py --config experiments/test.json`
4. Than you can see the results by running `analyze_results.py`:

```
$ python analyze_results.py -f none test
        dataset  flag  recall@20  recall@50  ndcg@100  training_time  std_recall@20  std_recall@50  std_ndcg@100  std_time  n_experiments
0        amazon  none   0.095107   0.110732  0.087584            0.0            NaN            NaN           NaN       NaN              1
1  amazon-small  test   0.026373   0.052412  0.025858            0.0            NaN            NaN           NaN       NaN              1
```

To use data in your own script please import `datasets.utils` and `config.config`. Then you can get the data like this:

```python
from datasets.utils import *
from config import config

dataset, params = config["amazon"]
dataset.load_interactions(**params)

X = get_sparse_matrix_from_dataframe(dataset.train_interactions)
```

Evaluation is done through `recpack`, models should have the `predict_df` method:

```python

class SomeModel:
  ...
  def predict_df(self, df: pd.DataFrame, n:int, batch_size:int = 1000):
    """
    This method recieves a dataframe with interactions and returns top_n predictions for every user as a dataframe
    (note that model can work with different number of items than evaluator)
    """
```

To evaluate a model you should do something like this:

```python
test_evaluator = Evaluation(dataset, "test")
test_df_preds = model.predict_df(test_evaluator.test_src)
test_results=test_evaluator(test_df_preds)
```

`test_results` is dictionary of {'metric_name':metric_value}




  

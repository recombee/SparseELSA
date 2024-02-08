# SparseELSA
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Forked from [Recommender System Datasets (and Baselines)](https://github.com/zombak79/recsysdata)

1. Download datasets. For example, for Movielens20m dataset go to datasets/ml20m folder and run `download_ml20m` script.
2. Install dependencies `pip install -r requirements.txt`
3. Run desired experiment. For example for Sparse ELSA run `python experiment_elsa.py --dataset ml20m --validation false --factors 800 --batch_size 1024 --model_strategy super_sparse --max_output 20000 --scheduler none --epochs 10 --tuning False --flag elsa_ml20m_test`
4. Run `python analyze_results.py -f elsa_ml20m_test` to see results of the experiment:
```
  dataset             flag  recall@20  recall@50  ndcg@100  training_time  std_recall@20  std_recall@50  std_ndcg@100  std_time  n_experiments
0   ml20m  elsa_ml20m_test   0.391659   0.528692  0.430469      32.278537            NaN            NaN           NaN       NaN              1
```



  

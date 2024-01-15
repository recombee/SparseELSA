from results import analyzer
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d","--dir", default="results", type=str, help="Dir where the results are stored")
parser.add_argument('-f','--flags', nargs='+', help='Set flag', default=[])

args = parser.parse_args([] if "__file__" not in globals() else None)

#print(args.flags)
#print(type(args.flags))
data = analyzer.get_raw_data(args.dir)
data = data[data.validation=="false"]
data['training_time'] = data['training_time'].apply(abs)
if len(args.flags)==0:
       flags = ["ease_test", "mf_test", "knn_test", "top_popular_test", "sparse_elsa_test", "dense_elsa_test"]
else:
       flags = args.flags

res_test1 = data[(data.flag.isin(flags))].groupby(["dataset","flag"])[['recall@20',
       'recall@50', 'ndcg@100', 'training_time']].mean().reset_index()
res_test2 = data[data.flag.isin(flags)].groupby(["dataset","flag"])[['recall@20',
       'recall@50', 'ndcg@100', 'training_time']].std().reset_index().iloc[:,2:]
res_test2.columns = ['std_recall@20',
       'std_recall@50', 'std_ndcg@100', 'std_time']
res_test3 = data[data.flag.isin(flags)].groupby(["dataset","flag"]).size().reset_index().iloc[:,2:]
res_test3.columns=["n_experiments"]

print(pd.concat([res_test1,res_test2,res_test3], axis=1).sort_values(["dataset","ndcg@100"]))
from results import analyzer
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d","--dir", default="results", type=str, help="Dir where the results are stored")
parser.add_argument('-f','--flags', nargs='+', help='Set flag', default=[])
parser.add_argument('-g','--groupby', nargs='+', help='Set groupbys', default=[])
parser.add_argument('-m','--mean', nargs='+', help='Set means', default=[])
parser.add_argument('-s','--std', nargs='+', help='Set stds', default=[])
parser.add_argument('-q','--query', default="seed==seed",type=str, help="query for limiting the results")
parser.add_argument('-t','--dataset', default="ml20m",type=str, help="Evaluated dataset.")

args = parser.parse_args([] if "__file__" not in globals() else None)

#print(args.flags)
#print(type(args.flags))

data = analyzer.get_raw_data(args.dir)

if len(args.groupby)==0:
       groupby = ["flag",'factors', 'batch_size', 'epochs','val_epoch','max_output']
else:
       groupby = args.groupby

if len(args.mean)==0:
       mean = ['val_recall@20', 'val_recall@50', 'val_ndcg@100', 'training_time']
else:
       mean = args.mean

if len(args.std)==0:
       std = ['val_recall@20', 'val_recall@50', 'val_ndcg@100', 'training_time']
else:
       std = args.std

if len(args.flags)==0:
       flags = ["ease_test", "mf_test", "knn_test", "top_popular_test", "sparse_elsa_test", "dense_elsa_test"]
else:
       flags = args.flags

data = data[data.dataset==args.dataset]
data = data.query(args.query)

cols = data.columns
data[cols] = data[cols].apply(pd.to_numeric, errors='ignore')

res_test1 = data[data.flag.isin(flags)].groupby(groupby)[mean].mean().reset_index()
res_test2 = data[data.flag.isin(flags)].groupby(groupby)[std].std().reset_index()[std]
res_test2.columns = [x.replace("val","std") if "val" in x else f"std_{x}" for x in std]
res_test3 = data[data.flag.isin(flags)].groupby(groupby)[mean].size().reset_index().iloc[:,-1:]
res_test3.columns=["n_experiments"]
pd.options.display.max_columns = None

print(pd.concat([res_test1,res_test2,res_test3], axis=1).sort_values([mean[-1]], ascending=False).head(50))
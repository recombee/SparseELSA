import pandas as pd
import os, pathlib
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False)

def fix_dup(iid, e):
    try:
        return e.loc[iid,"duplicates_first"]
    except KeyError:
        return iid
        
p = pathlib.Path(".")
dirs = [x for x in p.iterdir() if not x.is_file()]

for dir in dirs:
    cur_dir = str(dir)
    print(f"processing {cur_dir}")
    ratings_file = [x for x in os.listdir(dir) if x.split(".")[0].lower()==cur_dir.lower()][0]
    items_file = [x for x in os.listdir(dir) if cur_dir.lower() in x.lower() and x.split(".")[-1].lower()=="json"][0]

    interactions = pd.read_csv(os.path.join(dir, ratings_file), header=None)
    interactions.columns = ["item_id", "user_id", "rating", "timestamp"]

    items = pd.read_json(os.path.join(dir, items_file), lines=True)
    duplicates = pd.read_csv("duplicates.txt", header=None)
    duplicates.columns=["duplicates_txt"]
    duplicates["duplicates_array"] = duplicates["duplicates_txt"].apply(lambda x: x.split())

    duplicates["duplicates_first"]=duplicates.duplicates_array.parallel_apply(lambda x: x[0])
    duplicates = duplicates[["duplicates_first","duplicates_array"]]
    exploded_duplicates = duplicates.explode("duplicates_array")
    e=exploded_duplicates.set_index("duplicates_array")

    interactions["item_id"]=interactions["item_id"].parallel_apply(lambda x: fix_dup(x, e))

    interactions["item_id"] = interactions.item_id.astype('category')
    interactions["user_id"] = interactions.user_id.astype('category')

    interactions.to_csv(os.path.join(dir, "ratings.csv"))
    interactions.to_feather(os.path.join(dir, "ratings.feather"))

    items=items[items.asin.isin(interactions.item_id)]
    cols = items.columns
    desired_columns=["asin", "title", "category", "feature", "main_cat", "price"]
    for col in desired_columns:
        if col not in cols:
            print(f"adding {col} to columns")
            if col=="main_cat":
                items[col]=str(dir)
            else:
                items[col]=None
        else:
            print(f"column {col} ok")

    items=items[desired_columns]
    items=items.reset_index(drop=True)
    items.to_csv(os.path.join(dir, "items.csv"))
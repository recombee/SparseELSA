import pandas as pd
import os
from pathlib import Path, PurePath
from tqdm import tqdm

my_dir = Path('.')

dirs = [entry for entry in my_dir.iterdir() if entry.is_dir() and not str(entry)[0]=='.']

ratings_all = pd.concat([pd.read_feather(os.path.join(d, "ratings.feather")) for d in tqdm(dirs)])
ratings_all["item_id"] = ratings_all.item_id.astype('category')
ratings_all["user_id"] = ratings_all.user_id.astype('category')
ratings_all.user_id, ratings_all.item_id
ratings_all.reset_index(drop=True).to_feather("rating_all.feather")
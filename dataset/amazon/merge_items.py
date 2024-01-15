import pandas as pd
import os
from pathlib import Path, PurePath
from tqdm import tqdm

my_dir = Path('.')

dirs = [entry for entry in my_dir.iterdir() if entry.is_dir() and not str(entry)[0]=='.']

items_all = pd.concat([pd.read_csv(os.path.join(d, "items.csv")) for d in tqdm(dirs)])
items_all.reset_index(drop=True).to_feather("items_all.feather")
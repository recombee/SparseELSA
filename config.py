from datasets.utils import *


config = {
    "ml20m": 
    (
        Dataset("MovieLens20M"), 
        {            
            "filename" : "datasets/ml20m/ratings.csv",
            "item_id_name" : "movieId", 
            "user_id_name" : "userId",
            "value_name" : "rating",
            "timestamp_name" : "timestamp",
            "min_value_to_keep" : 4.,
            "user_min_support" : 5,
            "set_all_values_to" : 1.,
            "num_test_users": 10000,
            "random_state": 42,
            "load_previous_splits": False,
        }
    ),
    "netflix":
    (
        Dataset("NetflixPrize"),
        {
            "filename" : "datasets/netflix/netflix.csv",
            "value_name" : "rating",
            "timestamp_name" : "timestamp",
            "min_value_to_keep" : 4.,
            "user_min_support" : 5,
            "set_all_values_to" : 1.,
            "num_test_users": 40000,
            "random_state": 42,
            "load_previous_splits":False
        }
    ),
    "goodbooks":
    (
        Dataset("Goodbooks-10k"),
        {
            "raw_data" : """pd.read_csv("datasets/goodbooks/ratings.csv")""",
            "user_id_name":"user_id",
            "item_id_name":"book_id",
            "value_name":"rating",
            "min_value_to_keep" : 4.,
            "user_min_support" : 5,
            "item_min_support" : 1,
            "set_all_values_to" : 1.,
            "num_test_users": 2500,
            "random_state": 42,
            "load_previous_splits": False
        }
    ),
    "msd":
    (
        Dataset("MillionSongDataset"),
        {
            "raw_data" : """pd.read_csv("datasets/msd/msd_train_triplets.tsv", sep="\t", header=None)""",
            "user_id_name":0,
            "item_id_name":1,
            "value_name" : 2,
            "user_min_support" : 20,
            "item_min_support" : 200,
            "set_all_values_to" : 1.
        }
    ),
    "amazon":
    (
        Dataset("Amazon"),
        {
            "raw_data" : """pd.read_feather("datasets/amazon/rating_all.feather")""",
            "value_name" : "rating",
            "timestamp_name" : "timestamp",
            "min_value_to_keep" : 4.,
            "user_min_support" : 5,
            "item_min_support" : 5,
            "set_all_values_to" : 1.,
            "num_test_users": 50000,
            "random_state": 42,
            "max_steps": 1000
        }
    ),
    "amazon-small":
    (
        Dataset("Amazon"),
        {
            "raw_data" : """pd.read_feather("datasets/amazon/rating_all.feather")""",
            "value_name" : "rating",
            "timestamp_name" : "timestamp",
            "min_value_to_keep" : 4.,
            "user_min_support" : 5,
            "item_min_support" : 20,
            "set_all_values_to" : 1.,
            "num_test_users": 10000,
            "random_state": 42,
            "max_steps": 1000
        }
    ),
    "amazon-books":
    (
        Dataset("Amazon Books"),
        {
            "raw_data" : """pd.read_feather("datasets/amazon/books/interactions_pu20_pi20.feather")""",
            "value_name" : "value",
            "timestamp_name" : "timestamp",
            "min_value_to_keep" : 1.,
            "user_min_support" : 20,
            "item_min_support" : 20,
            "set_all_values_to" : 1.,
            "num_test_users": 10000,
            "random_state": 42,
            "max_steps": 1000
        }
    ),
}
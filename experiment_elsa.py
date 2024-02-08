import os
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Random seed")

parser.add_argument("--device", default=0, type=int, help="Default device to run on")

parser.add_argument("--factors", default=64, type=int, help="Number of model factors")
parser.add_argument("--batch_size", default=256, type=int, help="Batch size for model training")

parser.add_argument("--lr", default=.1, type=float, help="Learning rate for model training, only if scheduler is none")
parser.add_argument("--scheduler", default="none", type=str, help="Scheduler: LinearWarmup or none")
parser.add_argument("--init_lr", default=.001, type=float, help="starting lr, only if scheduler is not none")
parser.add_argument("--warmup_lr", default=.1, type=float, help="max warmup lr, only if scheduler is not none")
parser.add_argument("--target_lr", default=.001, type=float, help="final lr, only if scheduler is not none")
parser.add_argument("--weight_decay", default=0., type=float, help="weight decay for l2 regularozation of weights during training")

parser.add_argument("--epochs", default=10, type=int, help="Total epochs of model training")
parser.add_argument("--warmup_epochs", default=2, type=int, help="Number of epochs warming up during model training")
parser.add_argument("--decay_epochs", default=2, type=int, help="Number of epochs decaying during model training")

parser.add_argument("--model", default="elsa", type=str, help="model: [elsa]")
parser.add_argument("--model_strategy", default="dense", type=str, help="model: [dense, sparse, super_sparse]")

parser.add_argument("--dataset", default="", type=str, help="dataset. For list of datasets leave this parameter blank.")
parser.add_argument("--validation", default="false", type=str, help="Use validation split: true/false")

parser.add_argument("--pu", default=1, type=int, help="User pruning aplied on training data.")
parser.add_argument("--pi", default=1, type=int, help="Item pruning aplied on training data.")

parser.add_argument("--max_output", default=None, type=int, help="Max number of items on output for super sparse method.")

parser.add_argument("--shuffle", default=True, type=bool, help="shuffle order of users before each epoch")
parser.add_argument("--workers", default=0, type=int, help="Number of workers for dataloader.")
parser.add_argument("--use_multiprocessing", default=False, type=bool, help="use multipressing for dataloader")
parser.add_argument("--max_queue_size", default=20, type=int, help="maximum size of que of the data for dataloader")

parser.add_argument("--flag", default="none", type=str, help="flag for distinction of experiments, default none")


args = parser.parse_args([] if "__file__" not in globals() else None)

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.device}"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import math 
import keras

from ptELSA import *
from datasets.utils import *

from optimizers import NadamS
from schedules import LinearWarmup
from datasets.pydatasets import BasicRecSysDataset, PredictDfRecSysDataset, SparseRecSysDataset, SparseRecSysDatasetWithNegatives

from layers import LayerELSA, LayerANNA
from models import KerasANNA, KerasELSA, SparseKerasELSA, SparseKerasANNA, NMSE

from config import config

from time import time

if __name__ == "__main__":
    folder = f"results/{str(pd.Timestamp('today'))} {9*int(1e6)+np.random.randint(999999)}".replace(" ", "_")
    if not os.path.exists(folder):
        os.makedirs(folder)
    vargs = vars(args)
    vargs["cuda_or_cpu"]=DEVICE
    pd.Series(vargs).to_csv(f"{folder}/setup.csv")
    print(folder)
    torch.manual_seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    try:
        assert args.dataset in config.keys()
    except:
        print(f"Dataset must be one of {list(config.keys())}.")
        raise
    
    dataset, params = config[args.dataset]
    params['random_state'] = args.seed
    print(f"Loding dataset {args.dataset} with params {params}")
    dataset.load_interactions(**params)
    print(dataset)
    
    if args.validation == "true":
        print("creating validation evaluator")
        val_evaluator = Evaluation(dataset, "validation")
        df = fast_pruning(dataset.train_interactions, args.pu,args.pi)
    else:
        df = fast_pruning(dataset.full_train_interactions, args.pu,args.pi)
    
    X = get_sparse_matrix_from_dataframe(df)
    
    print(f"Interaction matrix: {repr(X)}")
    
    print("creating test evaluator")
    test_evaluator = Evaluation(dataset, "test")
    
    print()
    print(f"Creating model {args.model} with strategy {args.model_strategy}")
    
    

    if args.scheduler == "LinearWarmup":
        lr = LinearWarmup(
            warmup_steps=len(data_loader)*args.warmup_epochs,
            decay_steps=len(data_loader)*args.decay_epochs,
            starting_lr=args.init_lr, 
            warmup_lr=args.warmup_lr, 
            final_lr=args.target_lr, 
        )
    else:
        print(f"Using constant lr of {args.lr}")
        lr = args.lr
    
    # ELSA SECTION
    if args.model == "elsa" and args.model_strategy == "dense":
        data_loader = BasicRecSysDataset(X, args.batch_size, shuffle=args.shuffle, workers=args.workers, use_multiprocessing=args.use_multiprocessing, max_queue_size=args.max_queue_size)
        model = KerasELSA(X.shape[1], args.factors, df.item_id.cat.categories, device=DEVICE)
        model.to(DEVICE)
        model.compile(optimizer=NadamS(learning_rate=lr, weight_decay=args.weight_decay), loss=NMSE, metrics=[keras.metrics.CosineSimilarity()])
        model.train_step(data_loader[0])
    if args.model == "elsa" and args.model_strategy == "sparse":
        data_loader = SparseRecSysDataset(X, args.batch_size, shuffle=args.shuffle, workers=args.workers, use_multiprocessing=args.use_multiprocessing, max_queue_size=args.max_queue_size)
        model = SparseKerasELSA(X.shape[1], args.factors, df.item_id.cat.categories, device=DEVICE)
        model.to(DEVICE)
        model.compile(optimizer=NadamS(learning_rate=lr, weight_decay=args.weight_decay), loss=NMSE, metrics=[keras.metrics.CosineSimilarity()])
        model.train_step(data_loader[0])
    if args.model == "elsa" and args.model_strategy == "super_sparse":
        data_loader = SparseRecSysDatasetWithNegatives(X, device=DEVICE, batch_size=args.batch_size, shuffle=args.shuffle, workers=args.workers, use_multiprocessing=args.use_multiprocessing, max_queue_size=args.max_queue_size, max_output=args.max_output)
        model = SparseKerasELSA(X.shape[1], args.factors, df.item_id.cat.categories, device=DEVICE)
        model.to(DEVICE)
        model.compile(optimizer=NadamS(learning_rate=lr, weight_decay=args.weight_decay), loss=NMSE, metrics=[keras.metrics.CosineSimilarity()])
        model.train_step(data_loader[0])
    fits = []
    val_logs = []
    train_time=0
    if args.validation == "true": 
        for i in range(args.epochs):
            print(i)
            start = time()
            f = model.fit(data_loader, epochs=1, verbose=2)
            train_epoch_time = time()-start
            train_time+=train_epoch_time
            fits.append(f)
            val_df_preds = model.predict_df(val_evaluator.test_src)
            val_results=val_evaluator(val_df_preds)
            print(val_results)
            val_logs.append(val_results)
        dff = pd.DataFrame(val_logs)
        dff["epoch"] = np.arange(dff.shape[0])+1
        dff[list(dff.columns[-1:])+list(dff.columns[:-1])]
        dff.to_csv(f"{folder}/val_logs.csv")
        print("val_logs file written")
    else:
        start = time()
        f = model.fit(data_loader, epochs=args.epochs, verbose=2)
        train_time = time()-start
        fits.append(f)
        
    df_preds = model.predict_df(test_evaluator.test_src)
    results=test_evaluator(df_preds)
    
    print(results)
    ks = list(f.history.keys())    
    dc = {k:np.array([(f.history[k]) for f in fits]).flatten() for k in ks}
    dc["epoch"] = np.arange(len(dc[list(dc.keys())[0]]))+1
    df = pd.DataFrame(dc)
    df[list(df.columns[-1:])+list(df.columns[:-1])]
    
    df.to_csv(f"{folder}/history.csv")
    print("history file written")
    
    pd.Series(results).to_csv(f"{folder}/result.csv")
    print("results file written")
    
    pd.Series(train_time).to_csv(f"{folder}/timer.csv")
    print("timer written")
    
    
    

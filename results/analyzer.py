import pandas as pd
import os

def check_dir(dir_name):
    try:
        g = []
        for _, _, filenames in os.walk(dir_name):
            g.extend(filenames)
        assert 'history.csv' in g
        assert 'result.csv' in g
        assert 'setup.csv' in g
        assert 'timer.csv' in g
        #assert 'val_logs.csv' in g
        return True
    except:
        return False
    

def get_data_dirs(parent_dir = "."):
    f = []
    for _, dirnames, _ in os.walk(parent_dir):
        f.extend(dirnames)

    f = [x for x in f if check_dir(os.path.join(parent_dir,x))]
    
    return f

def get_results(path):
    setup = pd.read_csv(os.path.join(path, "setup.csv"))
    setup = setup.set_index(setup[setup.columns[0]], drop=True)["0"].to_frame().T
    result = pd.read_csv(os.path.join(path, "result.csv"))
    result = result.set_index(result[result.columns[0]], drop=True)["0"].to_frame().T
    timer = pd.read_csv(os.path.join(path, "timer.csv"))
    timer = timer.set_index(timer[timer.columns[0]], drop=True)["0"].to_frame().T
    timer.columns=(["training_time"])
    val_logs = pd.read_csv(os.path.join(path, "val_logs.csv")).iloc[:,1:]
    return pd.merge(how="cross", left=pd.concat([setup, timer], axis=1), right=val_logs)

def get_raw_data(parent_dir="."):
    f = get_data_dirs(parent_dir)

    return pd.concat([get_results(x) for x in f])

def get_results(path):
    setup = pd.read_csv(os.path.join(path, "setup.csv"))
    setup = setup.set_index(setup[setup.columns[0]], drop=True)["0"].to_frame().T
    result = pd.read_csv(os.path.join(path, "result.csv"))
    result = result.set_index(result[result.columns[0]], drop=True)["0"].to_frame().T
    timer = pd.read_csv(os.path.join(path, "timer.csv"))
    timer = timer.set_index(timer[timer.columns[0]], drop=True)["0"].to_frame().T
    timer.columns=(["training_time"])
    try:
        val_logs = pd.read_csv(os.path.join(path, "val_logs.csv")).iloc[:,1:]
        val_logs.columns=[f"val_{x}" for x in val_logs.columns]
        if len(val_logs)==0:
            raise FileNotFoundError
        ret=pd.merge(how="cross", left=pd.concat([setup, timer, result], axis=1), right=val_logs)
        ret["dir"]=path
        #print(f"{path} val")
        return ret
    except FileNotFoundError:
        ret=pd.concat([setup, timer, result], axis=1)
        ret["dir"]=path
        #print(f"{path} test")
        return ret


def get_raw_data(parent_dir=".", already_scanned=[]):
    f = get_data_dirs(parent_dir)
    #print(f)
    #print(pd.Series([str(os.path.join(parent_dir, x)) for x in f]))
    data = pd.concat([get_results(str(os.path.join(parent_dir, x))) for x in f if str(x) not in already_scanned])
    #data.to_feather(os.path.join(parent_dir, "data.feather"))
    return data

def get_data(parent_dir="."):
    cached_data = pd.read_csv(os.path.join(parent_dir, "data.csv"))
    #print(cached_data["dir"])
    try:
        new_data = get_raw_data(parent_dir=parent_dir, already_scanned=cached_data["dir"].to_list())
        data = pd.concat([cached_data, new_data]).reset_index(drop=True)
        data.to_csv(os.path.join(parent_dir, "data.csv"), index=False)
    except ValueError:
        data = cached_data
    return data 

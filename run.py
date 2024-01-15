import argparse
import json
import subprocess
import os
import itertools

parser = argparse.ArgumentParser()

parser.add_argument("--config", default="", type=str, help="Gridsearch config file")
args = parser.parse_args([] if "__file__" not in globals() else None)

if __name__ == "__main__":
    with open(args.config) as jsonfile:
        data = json.load(jsonfile)

    print(data)
    for script in list(data.keys()):

        list_data = {}
        static_data = {}

        for key in data[script].keys():
            if isinstance(data[script][key], list):
                list_data[key]=data[script][key]
            else:
                static_data[key]=data[script][key]

        dd={}

        for other_key in list_data.keys():
            ll=[]
            for l in list_data[other_key]:
                ll.append((other_key,l))
            dd[other_key]=ll

        list(dd.values())
        res = [list(x) for x in list(itertools.product(*list(dd.values())))]

        for r in res:
            for static_key, static_val in static_data.items():
                r.append((static_key, static_val))

        
        for parset in list(set([tuple(set(x)) for x in res])):
            print(parset)
            out = subprocess.check_output(
                [
                    'python', script,
                ] + [a for b in parset for a in b]
            )
            dirname=[x for x in out.decode("utf-8").split("\n") if x[:len("results/")]=="results/"][0]
            
            with open(os.path.join(dirname, 'stdout.log'), 'w') as f:
                f.write(out.decode("utf-8"))

            print(out)
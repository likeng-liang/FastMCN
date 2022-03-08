import csv
import pickle
from datetime import datetime
from itertools import chain, count
import torch
import numpy as np


def call_func(x, func):
    return func(x)


def call_method(lib, config):
    method = getattr(lib, config["name"])(**config["args"])
    return method


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def load_by_name(proc_path: str, obj_name: str):
    with open(f"{proc_path}/{obj_name}.pkl", "rb") as f:
        return pickle.load(f)


def mkdir_if_not_exists(path):
    if not path.exists():
        path.mkdir(parents=True)


def log_config(log_file: str, primary_key, config: dict, arg_keys: list):
    name = " | ".join(
        ["## " + primary_key, "name", f"\"{config['name']}\""]
    )
    args = [
        " | ".join(["## " + primary_key, k, f"\"{config['args'][k]}\""])
        for k in arg_keys
    ]
    content = "\n".join([name] + args) + "\n"
    with open(log_file, "a") as f:
        f.write(content)


def log(log_file, primary_key, metrics):
    with open(log_file, "a", newline='') as f:
        logs = [
            {**primary_key, "var": k, "value": v} for k, v in metrics.items()
        ]
        csv_writer = csv.DictWriter(f, list(logs[0].keys()), dialect="unix")
        csv_writer.writerows(logs)


def sparse_n_word(n_word):
    n_word_sum = n_word.sum()
    n_word_sp_idx = torch.stack(
        [
            torch.cat(
                [torch.tensor([idx] * n) for idx, n in zip(count(0), n_word)]
            ),
            torch.tensor(list(range(n_word_sum))),
        ]
    )
    n_word_sp_value = torch.ones(n_word_sum)
    n_word_sp_size = [n_word.shape[0], n_word_sum]
    n_word_spm_arg = {
        "indices": n_word_sp_idx,
        "values": n_word_sp_value,
        "size": n_word_sp_size,
    }
    return n_word_spm_arg


def prepare_eval_data(data, device="cuda"):
    word_code, name_code, n_word, name, label = zip(
        *[
            [
                dt[k]
                # fmt: off
                for k in ["word_code", "name_code", "n_word", "name", "concept"]
                # fmt: on
            ]
            for dt in chain(*data.values())
        ]
    )
    word_code = torch.tensor(list(chain(*word_code)), device=device) - 1
    n_word = np.array(n_word)
    n_word_spm_args = sparse_n_word(n_word)
    n_word_spm = torch.sparse.FloatTensor(**n_word_spm_args).to(device)
    name = [n.lower() for n in name]
    label = [l.lower() for l in label]
    data = {
        "word_code": word_code,
        "name_code": name_code,
        "n_word": n_word_spm,
        "name": name,
        "label": label,
    }
    return data

import sys
import csv
from importlib import import_module
from itertools import chain, product
from time import time
from collections import defaultdict
import torch
from torch.backends import cudnn
import numpy as np
from tqdm import trange
import module
from utils.func import call_method, now, log
from utils import dataset
from config.api import args

config_name = args.config
config_vars = import_module(f"config.{config_name}")
# config_var_names = [i for i in config_vars.__dir__() if not i.startswith("__")]

for n in config_vars.global_vars:
    globals()[n] = getattr(config_vars, n)


all_char_code, all_char_code_len, split_n = train_ds.split_char_code(
    all_char_code, train_ds.char_code_max_lens
)
all_char_code = all_char_code.to(DEVICE)
all_char_code_len = all_char_code_len.long()

train, test, dev = [eval_data[i] for i in ['train', "test", "dev"]]

for dt, k in product([train, test, dev], ["word_code", "n_word"]):
    dt[k] = dt[k].to(DEVICE)

if DEVICE == "cuda":
    model = model.cuda()

print(model)
model_size = np.sum([i.numel() for i in model.parameters()]) * 4 / 1024 / 1024
print(f"model_size: {model_size:.2f} Mib")

print(f"Everything will be stored at:\n{work_dir}")

formater = defaultdict(lambda: "%s: %.4f")
formater["step"] = "%s: %-4d"
best_metric = defaultdict(lambda: 0)

for step in range(N_ROUNDS):
    batch_start = time()
    cache = defaultdict(list)
    t = trange(STEP_PRE_ROUND, leave=False, ncols=120, ascii=True)
    for _ in t:
        forward_args, loss_args, n_concepts = train_ds.post_process(
            *yield_train_data()
        )
        token_ft = model(**forward_args)
        result = loss_func(token_ft, **loss_args)
        for param in model.parameters():
            param.grad = None

        result["loss"].backward()
        optimizer.step()
        result = {k: v.item() for k, v in result.items()}
        for key, value in result.items():
            cache[key].append(value)

        cache["n_concepts"].append(n_concepts)
    metric = {k: np.mean(v) for k, v in cache.items()}
    metric["step"] = step
    metric["eval_time"] = time() - batch_start
    print(" ".join([formater[k] % (k, v) for k, v in metric.items()]))
    primary_key = {
        "time": now(),
        "step": metric.pop("step"),
        "status": "train"
    }
    log(log_file, primary_key, metric)

    model = model.eval()
    cold_start = time()
    with torch.no_grad():
        token_ft = model.eval_encode_token(all_char_code, all_char_code_len)
        torch.cuda.empty_cache()
    train_mention_ft = model.eval_get_ft(token_ft, train['word_code'], train['n_word'])
    train_mention_ft = train_mention_ft.transpose(0, 1)
    warm_start = time()
    test_mention_ft = model.eval_get_ft(token_ft, test["word_code"], test["n_word"])
    test_pred = (test_mention_ft @ train_mention_ft).argmax(-1).tolist()
    eval_end = time()
    cold_start_time, warm_start_time = [eval_end - t for t in [cold_start, warm_start]]
    test_acc = np.mean([l == train["label"][p]
                        for l, p in zip(test["label"], test_pred)])
    dev_mention_ft = model.eval_get_ft(token_ft, dev["word_code"], dev["n_word"])
    dev_pred = (dev_mention_ft @ train_mention_ft).argmax(-1).tolist()
    dev_acc = np.mean([l == train["label"][p]
                       for l, p in zip(dev["label"], dev_pred)])
    eval_metric = {
        "cold_start": cold_start_time,
        "warm_start": warm_start_time,
        "test_acc": test_acc,
        "dev_acc": dev_acc
    }
    primary_key['status'] = "eval"
    log(log_file, primary_key, eval_metric)
    print(
        "\033[0;34m" +
        " ".join([formater[k] % (k, v) for k, v in eval_metric.items()])
        + "\033[0m"
    )

    if dev_acc > best_metric["best_dev_acc"]:
        best_metric["best_test_acc"] = test_acc
        best_metric["best_dev_acc"] = dev_acc
        best_metric["best_step"] = step
        content = "\n".join([
            "\t".join([n, l, train["label"][p]])
            for p, n, l in zip(test_pred, test["name"], test["label"])
        ])
        with open(f"{work_dir}/best_dev_acc_pred.tsv", "w") as f:
            f.write(content)
        # torch.save(model, f"{work_dir}/best_model.pyt")
    if test_acc > best_metric["highest_test_acc"]:
        best_metric["highest_test_acc"] = test_acc
        best_metric["highest_test_acc_step"] = step
    if (step + 1) % 10 == 0:
        primary_key['status'] = "best"
        log(log_file, primary_key, best_metric)
        print(
            "\033[1;31m" +
            " ".join([formater[k] % (k, v) for k, v in best_metric.items()]) +
            "\033[0m"
        )
        del token_ft, train_mention_ft, test_mention_ft, dev_mention_ft
        torch.cuda.empty_cache()
        model = model.train()

# torch.save(model, f"{work_dir}/end_model.pyt")
# torch.save(optimizer, f"{work_dir}/end_optimizer.pyt")

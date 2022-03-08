import os
import json
import random
from pathlib import Path
import torch
import numpy as np
from config.char2token2mention_default_config import *
from utils.func import *
import module


WINDOW = args.window
PRETRAIN_EPOCH = args.pretrain_epoch

train_ds = call_method(dataset, dataset_config)
yield_train_data = dataset.YieldData(
    train_ds, **{"num_workers": 4, "pin_memory": True}
)

encoder_config['name'] = "SVTransformerBPETokenV7Fast"
encoder_config['args']['n_head'] = args.n_head
encoder_config['args']['max_len'] = MAX_CHARS
encoder_config['args']['mlp_config'] = [
    [16, "tanh", True],
    [1, "tanh", True]
]

model_config = {
    "name": "Char2Token2Mention",
    "args": {"encoder_config": encoder_config},
}

char_code_max_lens = torch.tensor([0, 8, MAX_CHARS]).unsqueeze(-1)
yield_train_data.data_loader.dataset.char_code_max_lens = char_code_max_lens
dataset_config['name'] = "Char2Token2MentionDS"
dataset_config["args"]["char_code_max_lens"] = char_code_max_lens.tolist()

model = call_method(module.model, model_config)


home_path = os.path.expanduser("~")

vocab_path = f"{home_path}/data/pubmed_abstracts/ByteLevelBPE_uncase_131072words/sorted_vocab.json"
with open(vocab_path, "r") as f:
    sort_vocab = json.load(f)

char_map_new = {k: v for k, v in sort_vocab.items() if len(k) == 1}
char_map_old = load_by_name(DATA_PATH, "char_map")
char_map = {v: char_map_new[k] for k, v in char_map_old.items() if k in char_map_new}

pretrained_model_path = f"{home_path}/data/pubmed_abstracts/train/bpe_131072_uncase/MaskWordSVTV3_2/SmallFileDSV3_window{WINDOW}/SVTransformerBPETokenV7Fast/bs512_emb{args.emb_dim}_sentlen_32/"
sd = torch.load(f"{pretrained_model_path}/meta_module_epoch_{args.pretrain_epoch}.pyt").state_dict()
new_char_emb = torch.rand([encoder_config['args']['vocab_size'], args.emb_dim]) * 2 - 1
for k, v in char_map.items():
    new_char_emb[k] = sd['char'][v]

sd['char'] = sd['char_emb.weight'] = new_char_emb
model.token_encoder.load_state_dict(sd)
model = model.to(DEVICE)

optim_config["args"]["lr"] = args.lr
optim_config["args"]["params"] = model.parameters()

optimizer = call_method(torch.optim, optim_config)

work_dir = "/".join(
    [
        "./train",
        DATA,
        f"/{model_config['name']}/{encoder_config['name']}_pretrain_window{WINDOW}_epoch{PRETRAIN_EPOCH}",
        "_".join(
            [
                f"bs{BATCH_SIZE}",
                f"head{encoder_config['args']['n_head']}",
                f"emb{encoder_config['args']['emb_dim']}",
                f"lr{args.lr}",
                f"seed{args.seed}",
            ]
        ),
    ]
)

mkdir_if_not_exists(Path(work_dir))

# fmt: off
global_vars = [
    "model", "loss_func", "optimizer", "train_ds", "work_dir",
    "yield_train_data", "SEED", "DEVICE", "STEP_PRE_ROUND", "N_ROUNDS",
    "log_file", "start_time", "all_char_code", "train_data", "eval_data"
]
# fmt: on

start_time = now().split(".")[0]
log_file = work_dir + f"/{start_time}.log"

log_config(
    log_file,
    f"{start_time} | dataset_config",
    dataset_config,
    ["batch_size", "max_size_for_each_concept", "char_code_max_lens"],
)

log_config(
    log_file,
    f"{start_time} | model_config",
    model_config,
    [],
)

log_config(
    log_file,
    f"{start_time} | encoder_config",
    encoder_config,
    list(encoder_config["args"].keys()),
)

log_config(
    log_file,
    f"{start_time} | optim_config",
    optim_config,
    ["lr", "weight_decay"],
)

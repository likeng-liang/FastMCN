from config.api import args
from functools import partial
import torch
import utils.loss_func
from utils.func import load_by_name, call_method
from utils import dataset

DATA = args.data
DATA_PATH = f"./proc_data/{DATA}/"

train_data = load_by_name(DATA_PATH, "train_data")
eval_data = load_by_name(DATA_PATH, "eval_data")
all_char_code = load_by_name(DATA_PATH, "all_char_code")
info = load_by_name(DATA_PATH, "info")

SEED = args.seed

STEP_PRE_ROUND = 100
N_ROUNDS = args.n_round

MAX_CHARS = info["max length of word"]
VOCAB_SIZE = info["char map size"] + 1
SUB_BATCH_SIZE = 512
BATCH_SIZE = args.batch_size
DEVICE = "cuda"
# DEVICE = "cpu"

dataset_config = {
    "name": "Char2Token2MentionDS",
    "args": {
        "data": train_data,
        "batch_size": BATCH_SIZE,
        "max_size_for_each_concept": 2,
        "char_code_max_lens": torch.tensor([0, MAX_CHARS]).unsqueeze(-1),
        "device": DEVICE
    },
}

encoder_config = {
    "args": {
        "vocab_size": VOCAB_SIZE,
        "emb_dim": args.emb_dim,
        "dropout": 0.1,
        "device": DEVICE
    }
}

loss_func = partial(
    utils.loss_func.mcl,
    eye_mask=(torch.eye(BATCH_SIZE + 10) == 0).to(DEVICE),
    eps=torch.tensor(1e-4).float().to(DEVICE),
)

optim_config = {"name": "Adam", "args": {"lr": 1e-5, "weight_decay": 5e-5}}

import random
from pathlib import Path
import torch
import numpy as np
from config.char2token2mention_default_config import *
from utils.func import *
import module


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

train_ds = call_method(dataset, dataset_config)
yield_train_data = dataset.YieldData(
    train_ds, **{"num_workers": 4, "pin_memory": True}
)

encoder_config['name'] = "SVTransformerBPETokenV7Fast"
encoder_config['args']['n_head'] = args.n_head
encoder_config['args']['max_len'] = MAX_CHARS
encoder_config['args']['mlp_config'] = [
    [16, "tanh", False],
    [1, "tanh", False]
]

model_config = {
    "name": "Char2Token2Mention",
    "args": {"encoder_config": encoder_config},
}

char_code_max_lens = torch.tensor([0, 8, MAX_CHARS]).unsqueeze(-1)
yield_train_data.data_loader.dataset.char_code_max_lens = char_code_max_lens
dataset_config["args"]["char_code_max_lens"] = char_code_max_lens.tolist()

model = call_method(module.model, model_config)

optim_config["args"]["params"] = model.parameters()
optim_config["args"]["lr"] = args.lr

optimizer = call_method(torch.optim, optim_config)

work_dir = "/".join(
    [
        "./train",
        DATA,
        f"/{model_config['name']}/{encoder_config['name']}",
        "_".join(
            [
                f"bs{BATCH_SIZE}",
                f"head{encoder_config['args']['n_head']}",
                f"emb{encoder_config['args']['emb_dim']}",
                f"seed{args.seed}_{DEVICE}",
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

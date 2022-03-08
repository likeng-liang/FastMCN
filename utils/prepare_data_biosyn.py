import re
import sys
import pickle
import json
import csv
import logging
import torch
from pathlib import Path
from functools import reduce
from itertools import chain, count
from collections import defaultdict
from argparse import ArgumentParser
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from func import prepare_eval_data


parser = ArgumentParser()
parser.add_argument(
    "--input", "-i", default="../proc_data/ncbi_biosyn.csv"
)
parser.add_argument("--output", "-o", default="../proc_data/test/")
parser.add_argument("--charset", "-c", default=None)
parser.add_argument(
    "--size", "-s", default=32768, type=int, help="max size of vocabulary"
)
args = parser.parse_args()

input_file = args.input
proc_path = args.output
vocab_size = args.size
char_map_path = args.charset
log_file = f"{proc_path}/log.txt"
log_format = "%(asctime)s | %(message)35s | %(value)s"

info = {}

# fmt: off
stop_words = set()
dump_vars = [
    "train_data", "eval_data", "char_map", "word_map", "bpe_vocab_sorted",
    "bpe_words", "all_char_code", "info"
]
exclude_symbs = [" "]
# fmt: on


def set_logger(log_file, log_format):
    s_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file, mode="w")
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[s_handler, f_handler],
    )


def unique(obj, prefix="", verbose=True):
    s = set(chain(*obj))
    if verbose:
        print(f"{prefix}{len(s)}")
    return s


def split_to_words(x: str):
    return [w for w in __split_to_words(x) if w not in stop_words]


def dump_by_name(proc_path: str, obj_name: str):
    with open(f"{proc_path}/{obj_name}.pkl", "wb") as f:
        pickle.dump(globals()[obj_name], f)


def read_csv(file_path: str):
    with open(file_path, "r") as f:
        field_names = f.readline().replace("\n", "").split(",")
        dt = list(csv.DictReader(f, field_names))
    return dt


def get_names(data, stop_words):
    mentions = [i["mention"].lower() for i in data]

    logging.info("mentions", extra={"value": len(mentions)})
    info["mentions"] = len(mentions)
    logging.info("unique mentions", extra={"value": len(set(mentions))})
    info["unique mentions"] = len(set(mentions))
    concepts = set([c.lower() for i in data for c in i["concept"].split("|")])
    logging.info("unique concepts", extra={"value": len(concepts)})
    info["unique concepts"] = len(concepts)

    # name = mention
    names = unique([mentions], verbose=False)
    logging.info("unique names", extra={"value": len(names)})

    names = names - stop_words
    logging.info(
        "unique names without stop_words", extra={"value": len(names)}
    )
    info["unique names"] = len(names)
    return names


def train_tokenizer_and_build_vocab(name_path: list, output_dir: str):
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = Sequence(
        [
            # NFKC(),
            Lowercase()
        ]
    )
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=int(vocab_size), show_progress=True)
    tokenizer.train(trainer, name_path)
    logging.info(
        "vocabulary size", extra={"value": tokenizer.get_vocab_size()}
    )
    info["vocabulary size"] = tokenizer.get_vocab_size()

    tokenizer.model.save(proc_path)
    return tokenizer


def sort_vocab_and_update_tokenizer(proc_path, tokenizer):
    with open(f"{proc_path}/vocab.json", "r", encoding="utf8") as f:
        bpe_vocab = json.load(f)

    bpe_words = list(bpe_vocab.keys())
    bpe_words = sorted(bpe_words, key=lambda x: (len(x), x))
    bpe_vocab_sorted = {w: idx for w, idx in zip(bpe_words, count(1))}
    with open(f"{proc_path}/vocab_sorted.json", "w", encoding="utf8") as f:
        json.dump(bpe_vocab_sorted, f, indent=2, ensure_ascii=False)

    tokenizer.model = BPE.from_file(
        f"{proc_path}/vocab_sorted.json", f"{proc_path}/merges.txt"
    )
    return tokenizer, bpe_words, bpe_vocab_sorted


def build_char_and_word_maps(bpe_words):
    char_map = {k: v for k, v in zip(bpe_words, count(1)) if len(k) == 1}
    logging.info("char map size", extra={"value": len(char_map)})
    info["char map size"] = len(char_map)

    MAX_LEN_OF_WORD = max([len(w) for w in bpe_words])
    logging.info("max length of word", extra={"value": MAX_LEN_OF_WORD})
    info["max length of word"] = MAX_LEN_OF_WORD

    word_map = {
        k: [char_map[c] for c in k] + [0] * (MAX_LEN_OF_WORD - len(k))
        for k in bpe_words
    }
    return char_map, word_map


class RegSplit:
    """Split a string by regular expression
    Method: __init__, __call__
    """

    def __init__(self, pattern: str, exclude=[" "]):
        """pattern: string as regular expression"""
        import re

        self.pattern = pattern
        self.exclude = exclude
        self.spliter = re.compile(pattern)

    def __call__(self, string: str):
        finds = self.spliter.finditer(string)
        return [w[0] for w in finds if w[0] not in self.exclude]


class GenData:
    def __init__(self, name_bpe_words, char_map, word_map, bpe_vocab):
        self.__dict__.update(locals())
        self.append_funcs = {
            "train": self.train_append,
            "dev": self.test_dev_append,
            "test": self.test_dev_append,
        }

    def gen_result(self, name, concept):
        bpe_words = self.name_bpe_words[name]
        result = {
            "name": name,
            "word": bpe_words,
            "n_word": len(bpe_words),
            "char_code": [self.word_map[w] for w in bpe_words],
            "word_code": [self.bpe_vocab[w] for w in bpe_words],
            "name_code": [
                self.char_map[c] if c in self.char_map else 0 for c in name
            ],
            "concept": concept,
        }
        return result

    def append(self, dd, dt):
        dd = self.append_funcs[dt["set"]](dd, dt)
        return dd

    def train_append(self, dd, dt):
        concept = dt["concept"]
        mention = dt["mention"]
        if mention in self.name_bpe_words and ("|" not in concept):
            dd[dt["set"]][concept].append(self.gen_result(mention, concept))
        if len(dd[dt["set"]][concept]) > 1:
            return dd
        # if concept != mention and ("|" not in concept):
        #     dd[dt["set"]][concept].append(self.gen_result(concept, concept))
        return dd

    def test_dev_append(self, dd, dt):
        concept = dt["concept"]
        mention = dt["mention"]
        dd[dt["set"]]["data"].append(self.gen_result(mention, concept))
        return dd

    def __call__(self, data):
        dd = defaultdict(lambda: defaultdict(list))
        dd = reduce(self.append, data, dd)
        return dd


path = Path(proc_path)
if not path.exists():
    path.mkdir(parents=True)

set_logger(log_file, log_format)

__split_to_words = RegSplit(
    pattern="[a-zA-Z']+|[0-9]+|[^a-zA-Z0-9]",
    exclude=exclude_symbs,
)

all_data = read_csv(input_file)
data = [dt for dt in all_data if dt['set'] != "test"]

names = get_names(data, stop_words)
# remove some symbols in names
name_words = {n: " ".join(split_to_words(n)) for n in names}

with open(f"{proc_path}/names.txt", "w") as f:
    f.write("\n".join(list(name_words.values())))

tokenizer = train_tokenizer_and_build_vocab(
    [f"{proc_path}/names.txt"], proc_path
)
tokenizer, bpe_words, bpe_vocab_sorted = sort_vocab_and_update_tokenizer(
    proc_path, tokenizer
)

char_map, word_map = build_char_and_word_maps(bpe_words)

all_char_code = torch.tensor([word_map[i] for i in bpe_words])

name_bpe_words = {n: tokenizer.encode(w).tokens for n, w in name_words.items()}
MAX_LEN_OF_MENTION = max([len(v) for v in name_bpe_words.values()])
logging.info("max length of name", extra={"value": MAX_LEN_OF_MENTION})
info["max length of name"] = MAX_LEN_OF_MENTION

gen_data = GenData(name_bpe_words, char_map, word_map, bpe_vocab_sorted)
data_sets = gen_data(all_data)

train_data = dict(data_sets["train"])
eval_data = {k: prepare_eval_data(dict(v)) for k, v in data_sets.items()}

print(f"Dumping data at {proc_path}:")
for i in dump_vars:
    print(i)
    dump_by_name(proc_path, i)

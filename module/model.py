# Import


# [[file:~/Works/char2token2mention/module/model.org::*Import][Import:1]]
import torch
from torch import nn
from torch.nn import Module
import module.encoder
from utils.func import call_method
from torch.nn.functional import normalize

# Import:1 ends here

# Functions


# [[file:~/Works/char2token2mention/module/model.org::*Functions][Functions:1]]
def sort_inputs(inputs):
    lens = (inputs > 0).sum(-1)
    lens_sorted, lens_sort_idx = lens.sort(0, descending=True)
    _, lens_unsort_idx = torch.sort(lens_sort_idx, dim=0)
    inputs_sorted = inputs[lens_sort_idx]
    lens_sorted = lens_sorted.to("cpu")
    result = {
        "inputs_sorted": inputs_sorted,
        "lens_sorted": lens_sorted,
        "lens_unsort_idx": lens_unsort_idx,
    }
    return result


# Functions:1 ends here

# Char2Token2Mention


# [[file:~/Works/char2token2mention/module/model.org::*Char2Token2Mention][Char2Token2Mention:1]]
class Char2Token2Mention(Module):
    def __init__(self, encoder_config, eval_sub_batch_size=1024):
        super(Char2Token2Mention, self).__init__()
        self.__dict__.update(locals())
        self.token_encoder = call_method(module.encoder, encoder_config)
        if hasattr(self.token_encoder, "encode_token"):
            self.encode_token = self.token_encoder.encode_token
        if hasattr(self.token_encoder, "eval_encode_token"):
            self.eval_encode_token = self.token_encoder.eval_encode_token

    def _encode_token(self, code, lens):
        token_ft = self.token_encoder(code, lens)
        return token_ft

    def encode_token(self, char_code, char_len):
        token_ft = torch.cat(
            [
                self._encode_token(code, l)
                for code, l in zip(char_code, char_len)
            ],
            dim=0,
        )
        return token_ft

    def encode_mention(self, token_ft, token_code, token_spm):
        ft = token_ft.index_select(0, token_code)
        mention_ft = torch.spmm(token_spm, ft)
        return mention_ft

    def forward(self, char_code, char_len, token_code, token_spm):
        # char_len is sorted
        token_ft = self.encode_token(char_code, char_len)
        mention_ft = self.encode_mention(token_ft, token_code, token_spm)
        return mention_ft

    def eval_encode_token(self, char_code, char_len):
        token_ft = torch.cat(
            [
                self._encode_token(code, l)
                for code, l in zip(
                    char_code.split(self.eval_sub_batch_size),
                    char_len.split(self.eval_sub_batch_size),
                )
            ],
            dim=0,
        )
        return token_ft

    def eval_get_ft(self, token_ft, token_code, token_spm):
        ft = self.encode_mention(token_ft, token_code, token_spm)
        return normalize(ft)


# Char2Token2Mention:1 ends here

# Char2Token2MentionCE


# [[file:~/Works/char2token2mention/module/model.org::*Char2Token2MentionCE][Char2Token2MentionCE:1]]
class Char2Token2MentionCE(Char2Token2Mention):
    def __init__(self, encoder_config, n_classes, eval_sub_batch_size=1024):
        super(Char2Token2MentionCE, self).__init__(
            encoder_config, eval_sub_batch_size
        )
        self.__dict__.update(locals())
        self.bn = nn.LayerNorm(self.token_encoder.output_dim)
        self.linear = nn.Linear(self.token_encoder.output_dim, n_classes)

    def forward(self, char_code, char_len, token_code, token_spm):
        # char_len is sorted
        token_ft = self.encode_token(char_code, char_len)
        mention_ft = self.encode_mention(token_ft, token_code, token_spm)
        pred = self.linear(mention_ft)
        return pred

    def eval_get_ft(self, token_ft, token_code, token_spm):
        ft = self.encode_mention(token_ft, token_code, token_spm)
        pred = self.linear(ft)
        return pred


# Char2Token2MentionCE:1 ends here

# Char2Token2MentionCEL


# [[file:~/Works/char2token2mention/module/model.org::*Char2Token2MentionCEL][Char2Token2MentionCEL:1]]
Char2Token2MentionCEL = Char2Token2Mention
# Char2Token2MentionCEL:1 ends here

# Char2Mention


# [[file:~/Works/char2token2mention/module/model.org::*Char2Mention][Char2Mention:1]]
class Char2Mention(Char2Token2Mention):
    def __init__(self, encoder_config, eval_sub_batch_size=512):
        super(Char2Mention, self).__init__(encoder_config, eval_sub_batch_size)

    def forward(self, name_code, name_len, unsort_idx):
        # name_len is sorted
        mention_ft = self.encode_token(name_code, name_len)
        mention_ft = mention_ft[unsort_idx]
        return mention_ft

    def __eval_get_ft(self, code, lens):
        with torch.no_grad():
            ft = self.token_encoder(code, lens)
        torch.cuda.empty_cache()
        return ft

    def eval_get_ft(self, code, lens):
        ft = torch.cat(
            [
                self.__eval_get_ft(c, l)
                for c, l in zip(
                    code.split(self.eval_sub_batch_size),
                    lens.split(self.eval_sub_batch_size),
                )
            ],
            dim=0,
        )
        return normalize(ft)


# Char2Mention:1 ends here

# Char2MentionCE


# [[file:~/Works/char2token2mention/module/model.org::*Char2MentionCE][Char2MentionCE:1]]
class Char2MentionCE(Char2Mention):
    def __init__(self, encoder_config, n_classes, eval_sub_batch_size=512):
        super(Char2MentionCE, self).__init__(
            encoder_config, eval_sub_batch_size
        )
        self.__dict__.update(locals())
        self.linear = nn.Linear(self.token_encoder.output_dim, n_classes)

    def forward(self, name_code, name_len, unsort_idx):
        # char_len is sorted
        mention_ft = self.encode_token(name_code, name_len)
        pred = self.linear(mention_ft)
        pred = pred[unsort_idx]
        return pred

    def __eval_get_ft(self, code, lens):
        with torch.no_grad():
            ft = self.token_encoder(code, lens)
            pred = self.linear(ft)
        torch.cuda.empty_cache()
        return pred

    def eval_get_ft(self, code, lens):
        pred = torch.cat(
            [
                self.__eval_get_ft(c, l)
                for c, l in zip(
                    code.split(self.eval_sub_batch_size),
                    lens.split(self.eval_sub_batch_size),
                )
            ],
            dim=0,
        )
        return pred


# Char2MentionCE:1 ends here

# Char2MentionCEL


# [[file:~/Works/char2token2mention/module/model.org::*Char2MentionCEL][Char2MentionCEL:1]]
Char2MentionCEL = Char2Mention
# Char2MentionCEL:1 ends here

# Token2Mention


# [[file:~/Works/char2token2mention/module/model.org::*Token2Mention][Token2Mention:1]]
class Token2Mention(Char2Token2Mention):
    def __init__(self, encoder_config, eval_sub_batch_size=512):
        super(Token2Mention, self).__init__(
            encoder_config, eval_sub_batch_size
        )

    def forward(self, word_code, word_len, unsort_idx):
        # word_len is sorted
        mention_ft = self.encode_token(word_code, word_len)
        mention_ft = mention_ft[unsort_idx]
        return mention_ft

    def __eval_get_ft(self, code, lens):
        with torch.no_grad():
            ft = self.token_encoder(code, lens)
        torch.cuda.empty_cache()
        return ft

    def eval_get_ft(self, code, lens):
        ft = torch.cat(
            [
                self.__eval_get_ft(c, l)
                for c, l in zip(
                    code.split(self.eval_sub_batch_size),
                    lens.split(self.eval_sub_batch_size),
                )
            ],
            dim=0,
        )
        return normalize(ft)


# Token2Mention:1 ends here

# Token2MentionCE


# [[file:~/Works/char2token2mention/module/model.org::*Token2MentionCE][Token2MentionCE:1]]
class Token2MentionCE(Token2Mention):
    def __init__(self, encoder_config, n_classes, eval_sub_batch_size=512):
        super(Token2MentionCE, self).__init__(
            encoder_config, eval_sub_batch_size
        )
        self.__dict__.update(locals())
        self.linear = nn.Linear(self.token_encoder.output_dim, n_classes)

    def forward(self, word_code, word_len, unsort_idx):
        # char_len is sorted
        mention_ft = self.encode_token(word_code, word_len)
        pred = self.linear(mention_ft)
        pred = pred[unsort_idx]
        return pred

    def __eval_get_ft(self, code, lens):
        with torch.no_grad():
            ft = self.token_encoder(code, lens)
            pred = self.linear(ft)
        torch.cuda.empty_cache()
        return pred

    def eval_get_ft(self, code, lens):
        pred = torch.cat(
            [
                self.__eval_get_ft(c, l)
                for c, l in zip(
                    code.split(self.eval_sub_batch_size),
                    lens.split(self.eval_sub_batch_size),
                )
            ],
            dim=0,
        )
        return pred


# Token2MentionCE:1 ends here

# Token2MentionCEL


# [[file:~/Works/char2token2mention/module/model.org::*Token2MentionCEL][Token2MentionCEL:1]]
Token2MentionCEL = Token2Mention
# Token2MentionCEL:1 ends here

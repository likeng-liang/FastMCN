# Import


# [[file:~/Works/char2token2mention/module/encoder.org::*Import][Import:1]]
import torch
from torch import nn
from torch.nn import Module
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from functools import partial, reduce
from utils.func import call_func
# Import:1 ends here

# Function


# [[file:~/Works/char2token2mention/module/encoder.org::*Function][Function:1]]
def mean_without_zero(tensor, n_not_zero, dim):
    return tensor.sum(dim=dim) / n_not_zero
# Function:1 ends here

# Modules


# [[file:~/Works/char2token2mention/module/encoder.org::*Modules][Modules:1]]
class FeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.__dict__.update(locals())
        self.layers = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_in),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):
        outputs = x + self.layers(x)
        x = self.layer_norm(x)
        return x
# Modules:1 ends here

# [[file:~/Works/char2token2mention/module/encoder.org::*Modules][Modules:2]]
class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, activate_func, bias=True):
        """A linear layer and an activate function.
        input_dim: Int. The dimension of input.
        output_dim: Int.  The dimension of output.
        activate_func: str. Activate function which were defined at utils.activate.
        bias: Bool. bias argument for linear layer."""
        super().__init__()
        args = locals()
        args_names = [key for key in args if key != "self"]
        for n in args_names:
            setattr(self, n, args[n])
            # self.act_func = torch.jit.script(getattr(activate, activate_func)())
        self.act_func = getattr(torch, activate_func) if activate_func else self.id
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def id(self, inputs):
        return inputs

    def forward(self, inputs):
        return self.act_func(self.linear(inputs))
# Modules:2 ends here

# [[file:~/Works/char2token2mention/module/encoder.org::*Modules][Modules:3]]
class ComputeQKV(nn.Module):
    def __init__(self, n_head, word_vec_d, n_layer=3, bias=False):
        super().__init__()
        self.n_head = n_head
        self.word_vec_d = word_vec_d
        self.n_layer = n_layer
        self.bias = bias
        out_features = self.n_layer * self.n_head * self.word_vec_d
        self.linear = nn.Linear(word_vec_d, out_features, bias=self.bias)

    def forward(self, inputs):
        qkv_shape = [inputs.shape[0], self.n_layer, self.n_head, -1]
        return self.linear(inputs).reshape(qkv_shape)
# Modules:3 ends here

# BiLSTM


# [[file:~/Works/char2token2mention/module/encoder.org::*BiLSTM][BiLSTM:1]]
class BiLSTM(Module):
    def __init__(self, vocab_size, emb_dim, n_layer, dropout=0.1, device="cuda"):
        super().__init__()
        self.__dict__.update(locals())
        self.emb_layer = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim,
            num_layers=n_layer,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = emb_dim * 2

    def run_bilstm(self, emb, lens):
        packed_emb = pack_padded_sequence(
            input=emb, lengths=lens, batch_first=True
        )
        packed_out, _ = self.bilstm(packed_emb)
        output, _ = pad_packed_sequence(packed_out, batch_first=True)
        return output

    def forward(self, inputs, lens):
        idx = torch.arange(lens.shape[0] - 1, -1, -1, dtype=torch.long)
        inputs, lens = [i[idx] for i in [inputs, lens]]
        emb = self.emb_layer(inputs)
        output = self.run_bilstm(emb, lens)
        output = self.dropout(output)
        output = mean_without_zero(
            output, lens.unsqueeze(-1).to(self.device), dim=-2
        )
        output = output[idx]
        return output
# BiLSTM:1 ends here

# TextCNN


# [[file:~/Works/char2token2mention/module/encoder.org::*TextCNN][TextCNN:1]]
class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_filter, kernel_size=[2, 3, 4, 5], device="cuda"):
        super().__init__()
        self.__dict__.update(locals())
        self.emb_layer = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(emb_dim, n_filter, k)
            for k in kernel_size
        ])
        self.output_dim = n_filter * len(kernel_size)

    def conv1d_block(self, inputs, conv):
        output = conv(inputs)
        output, _ = output.max(-1)
        return output

    def forward(self, inputs, lens):
        # lens must be sorted.
        max_len = max(lens[-1], self.kernel_size[-1])
        inputs = inputs[:, :max_len]
        emb = self.emb_layer(inputs).transpose(-1, -2)
        output = torch.cat([self.conv1d_block(emb, c) for c in self.conv_layers],
                           dim=-1)
        return output
# TextCNN:1 ends here

# TextCNNV2


# [[file:~/Works/char2token2mention/module/encoder.org::*TextCNNV2][TextCNNV2:1]]
class TextCNNV2(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_filter, kernel_size=[2, 3, 4, 5], device="cuda"):
        super().__init__()
        self.__dict__.update(locals())
        self.emb_layer = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(emb_dim, n_filter, k)
            for k in kernel_size
        ])
        self.bns = nn.ModuleList([
            nn.LayerNorm(self.n_filter)
            for k in kernel_size
        ])
        self.output_dim = n_filter * len(kernel_size)

    def conv1d_block(self, inputs, conv, bn):
        output = conv(inputs)
        output, _ = output.max(-1)
        output = bn(output)
        return output

    def forward(self, inputs, lens):
        # lens must be sorted.
        max_len = max(lens[-1], self.kernel_size[-1])
        inputs = inputs[:, :max_len]
        emb = self.emb_layer(inputs).transpose(-1, -2)
        output = torch.cat([self.conv1d_block(emb, c, bn)
                            for c, bn in zip(self.conv_layers, self.bns)],
                       dim=-1)
        output = torch.tanh(output)
        return output
# TextCNNV2:1 ends here

# Transformer


# [[file:~/Works/char2token2mention/module/encoder.org::*Transformer][Transformer:1]]
def gen_pos_emb(emb_dim, max_len):
    emb_dim_half = emb_dim >> 1
    freq = torch.pow(torch.tensor([1e4]), -1 / emb_dim_half).repeat(
        [emb_dim_half]
    )
    freq[0] = 1.0
    freq = freq.cumprod(-1)
    position = torch.arange(0, max_len)
    phase = torch.einsum("i, j->ij", position, freq)
    pos_emb = torch.zeros([max_len, emb_dim])
    pos_emb[:, 0::2] = torch.sin(phase)
    pos_emb[:, 1::2] = torch.cos(phase)
    return pos_emb.unsqueeze(1)


class MultiHeadAttentionSum(nn.Module):
    def __init__(self, emb_dim, n_head, d_qkv=None):
        super().__init__()
        self.__dict__.update(locals())
        out_dim = emb * 3 if not d_qkv else d_qkv * n_head * 3
        self.qkv_linear = torch.nn.Linear(emb_dim, int(out_dim), bias=True)
        self.scaling = torch.pow(torch.tensor(emb_dim / n_head), -0.5)
        self.gn = torch.nn.GroupNorm(n_head, n_head)
        self.split_to_qkv = partial(self.split_to_qkv, n_head=n_head)
        self.compute_attn_weights = partial(
            self.compute_attn_weights, scaling=self.scaling
        )
        self.compute_attn_output = partial(
            self.compute_attn_output, n_head=n_head
        )

    @staticmethod
    def split_to_qkv(ft, max_len: int, bs: int, n_head: int):
        q, k, v = (
            ft.reshape(max_len, bs * n_head, -1)
            .transpose(0, 1)
            .chunk(3, dim=-1)
        )
        return q, k, v

    @staticmethod
    def compute_attn_weights(q, k, scaling):
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * scaling
        attn_mask = attn_weights == 0
        attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, -1)
        attn_weights = attn_weights.masked_fill(attn_mask, 0).sum(-2)
        return attn_weights

    @staticmethod
    def compute_attn_output(attn_weights, v, bs: int, n_head: int):
        attn_output = torch.einsum("bl, bld -> bd", attn_weights, v)
        attn_output = attn_output.reshape(bs, n_head, -1)
        return attn_output

    def forward(self, emb):
        max_len, bs = emb.shape[:-1]
        q, k, v = self.split_to_qkv(self.qkv_linear(emb), max_len, bs)
        attn_weights = self.compute_attn_weights(q, k)
        attn_output = self.compute_attn_output(attn_weights, v, bs)
        attn_output = reduce(call_func, [self.gn], attn_output)
        attn_output = attn_output.reshape(bs, -1)
        return attn_output

    def post_process(self, ft, emb, lens):
        return ft + emb.sum(0) / lens


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_head):
        super().__init__()
        self.__dict__.update(locals())
        self.qkv_linear = torch.nn.Linear(emb_dim, emb_dim * 3, bias=True)
        self.scaling = torch.pow(torch.tensor(emb_dim / n_head), -0.5)
        # self.gn = torch.nn.GroupNorm(n_head, n_head)
        self.split_to_qkv = partial(self.split_to_qkv, n_head=n_head)
        self.compute_attn_weights = partial(
            self.compute_attn_weights, scaling=self.scaling
        )

    @staticmethod
    def split_to_qkv(ft, max_len: int, bs: int, n_head: int):
        q, k, v = (
            ft.reshape(max_len, bs * n_head, -1)
            .transpose(0, 1)
            .chunk(3, dim=-1)
        )
        return q, k, v

    @staticmethod
    def compute_attn_weights(q, k, scaling):
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * scaling
        attn_mask = attn_weights == 0
        attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, -1)
        attn_weights = attn_weights.masked_fill(attn_mask, 0)
        return attn_weights

    @staticmethod
    def compute_attn_output(attn_weights, v):
        attn_output = torch.einsum("bal, bld -> bad", attn_weights, v)
        # max_len, batch * n_head, emb_dim / n_head
        attn_output = attn_output.transpose(0, 1)
        return attn_output

    def forward(self, emb):
        # emb: max_len, batch, emb_dim
        max_len, bs = emb.shape[:-1]
        q, k, v = self.split_to_qkv(self.qkv_linear(emb), max_len, bs)
        attn_weights = self.compute_attn_weights(q, k)
        # max_len, batch * n_head, emb_dim
        attn_output = self.compute_attn_output(attn_weights, v)
        # attn_output = reduce(call_func, [self.gn], attn_output)
        attn_output = attn_output.reshape(max_len, bs, -1)
        return attn_output

    def post_process(self, ft, emb, lens):
        return ft + emb


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self, emb_dim, n_head, mha, dropout=0.1
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.mha = mha(emb_dim, n_head)
        dim_feedforward = emb_dim * 2
        self.linear1 = nn.Linear(emb_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, emb_dim)

        self.norm1 = nn.LayerNorm(emb_dim)
        # self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, emb, lens):
        ft = self.mha(emb)
        emb = self.mha.post_process(ft, emb, lens)
        ft = reduce(
            call_func,
            [
                self.norm1,
                self.linear1,
                torch.tanh,
                self.linear2,
            ],
            emb,
        )
        # emb = self.norm2(emb + ft)
        emb = emb + ft
        return emb


class TransformerEncoder(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, n_head, max_len, dropout=0.1, device="cuda"
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.emb_layer = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb = gen_pos_emb(emb_dim, max_len).to(device)
        # self.layer1 = TransformerEncoderLayer(emb_dim, n_head, MultiHeadAttention)
        self.layer2 = TransformerEncoderLayer(emb_dim, n_head, MultiHeadAttentionSum)

    def forward(self, inputs, lens):
        # inputs is batch first
        bs = inputs.shape[0]
        # lens has been sorted
        max_len = lens[-1]
        inputs = inputs[:, :max_len]
        emb = self.emb_layer(inputs.T) + self.pos_emb[:max_len]
        lens = lens.unsqueeze(-1).to(self.device)
        # emb = self.layer1(emb, lens)
        emb = self.layer2(emb, lens)
        return emb


class MHA(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, n_head, max_len, dropout=0.1, device="cuda"
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.emb_layer = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb = gen_pos_emb(emb_dim, max_len).to(device)
        # self.layer1 = TransformerEncoderLayer(emb_dim, n_head, MultiHeadAttention)
        d_qkv = int(emb_dim / n_head)
        self.mha = MultiHeadAttentionSum(emb_dim, n_head, d_qkv=d_qkv)
        self.output_dim = d_qkv * n_head

    def forward(self, inputs, lens):
        # inputs is batch first
        bs = inputs.shape[0]
        # lens has been sorted
        max_len = lens[-1]
        inputs = inputs[:, :max_len]
        emb = self.emb_layer(inputs.T) + self.pos_emb[:max_len]
        lens = lens.unsqueeze(-1).to(self.device)
        ft = self.mha(emb)
        emb = torch.tanh(ft)
        return emb
# Transformer:1 ends here

# SVT


# [[file:~/Works/char2token2mention/module/encoder.org::*SVT][SVT:1]]
class SVTransformerBPETokenV7Fast(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        n_head,
        emb_dim,
        dropout,
        mlp_config,
        pad_idx=0,
        device="cuda",
        eval_sub_batch_size=512
    ):
        super().__init__()
        # attributes
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.mlp_config = mlp_config
        self.device = device
        self.pad_idx = pad_idx
        self.mha_dim = n_head
        self.char_emb = nn.Embedding(
            vocab_size, self.emb_dim, padding_idx=pad_idx
        ).to(device)
        self.char = self.char_emb.weight
        self.pos = self.get_pos(self.max_len, self.emb_dim).to(device)
        self.char_qkv_layer = ComputeQKV(self.n_head, self.emb_dim)
        self.pos_qkv_layer = ComputeQKV(self.n_head, self.emb_dim)
        # sdpa: scaled dot product attention
        self.sdpa_temperature = emb_dim ** 0.5
        self.sdpa_dropout = nn.Dropout(dropout)
        self.sdpa_bn_layer = nn.LayerNorm(emb_dim, eps=1e-5)
        # layers
        self.mlp_dims = [self.mha_dim] + [c[0] for c in mlp_config[:-1]]
        self.mlp = nn.Sequential(
            *[Dense(i, *c) for i, c in zip(self.mlp_dims, mlp_config)]
        )
        self.eval_sub_batch_size = eval_sub_batch_size

    def get_pos(self, max_len, emb_dim):
        emb_dim_half = emb_dim >> 1
        freq = torch.pow(torch.tensor([1e4]), -1 / emb_dim_half).repeat(
            [emb_dim_half]
        )
        freq[0] = 1.0
        freq = freq.cumprod(-1)
        position = torch.arange(0, max_len)
        phase = torch.einsum("i, j->ij", position, freq)
        pos_embedding = torch.zeros([max_len, emb_dim])
        pos_embedding[:, 0::2] = torch.sin(phase)
        pos_embedding[:, 1::2] = torch.cos(phase)
        return pos_embedding

    def compute_qkv(self):
        char_qkv = self.char_qkv_layer(self.char)
        pos_qkv = self.pos_qkv_layer(self.pos)
        return self.interactive_char_pos([char_qkv, pos_qkv])

    def interactive_char_pos(self, qkv):
        char_qkv, pos_qkv = qkv
        char_q, char_k, char_v = [
            char_qkv[:, i] for i in range(char_qkv.shape[1])
        ]
        pos_q, pos_k, pos_v = [pos_qkv[:, i] for i in range(pos_qkv.shape[1])]
        # (vocab_size, n_head, emb_dim) -> (n_head, vocab_size, emb_dim)
        char_q = char_q.transpose(0, 1)
        # (vocab_size, n_head, emb_dim) -> (n_head, emb_dim, vocab_size)
        char_k = char_k.permute(1, -1, 0)
        # n_head, vocab_size, vocab_size
        char_qk = (char_q @ char_k) / self.sdpa_temperature
        char_qk[:, :, self.pad_idx] = -1e9
        # (max_len, n_head, emb_dim) -> (n_head, max_len, emb_dim)
        pos_q = pos_q.transpose(0, 1)
        # (max_len, n_head, emb_dim) -> (n_head, emb_dim, max_len)
        pos_k = pos_k.permute(1, -1, 0)
        # (max_len, n_head, emb_dim) -> (1, n_head, max_len, emb_dim)
        pos_qk = (pos_q @ pos_k).unsqueeze(0) / self.sdpa_temperature
        # n_head, vocab_size, max_len
        char_pos_qk = (
            (char_q @ pos_k) + (pos_q @ char_k).transpose(-1, -2)
        ) / self.sdpa_temperature
        # max_len, n_head, emb_dim -> 1, n_head, max_len, emb_dim
        pos_v = pos_v.transpose(0, 1).unsqueeze(0)
        return char_qk, pos_qk, char_pos_qk, char_v, pos_v

    def scaled_dot_product_attention(self, inputs, not_mask, qkv, max_len):
        char_qk, pos_qk, char_pos_qk, char_v, pos_v = qkv
        # (n_head, vocab_size, vocab_size) -> (n_head, n_words, max_len, vocab_size)
        inputs_char_qk = char_qk[
            :,
            inputs,
        ]
        column_indexes = torch.einsum(
            "ij, ik -> ijk", torch.ones_like(inputs), inputs
        ).repeat([self.n_head, 1, 1, 1])
        # (n_head, n_words, max_len, vocab_size) -> (n_head, n_words, max_len, max_len)
        inputs_char_qk = inputs_char_qk.gather(-1, column_indexes)
        # (n_head, n_words, max_len, max_len) -> (n_words, n_head, max_len, max_len)
        inputs_char_qk = inputs_char_qk.transpose(0, 1)
        # (n_head, vocab_size, max_len) -> (n_words, n_head, max_len, max_len)
        inputs_char_pos_qk = char_pos_qk[:, inputs, :max_len].transpose(0, 1)
        # n_words, n_head, max_len, max_len
        inputs_sim = (
            inputs_char_qk
            + inputs_char_pos_qk
            + pos_qk[:, :, :max_len, :max_len]
        )
        # n_words, n_head, max_len, max_len
        inputs_attn = self.sdpa_dropout(
            nn.functional.softmax(inputs_sim, dim=-1)
        )
        # n_words, n_head, max_len
        not_mask_n_words = not_mask.sum(-1)
        inputs_attn = (not_mask @ inputs_attn).squeeze(-2) / not_mask_n_words
        # 1, n_head, max_len, emb_dim
        inputs_pos_v = pos_v[:, :, :max_len]
        # max_len, n_head, emb_dim -> n_words, n_head, max_len, emb_dim
        inputs_v = char_v[
            inputs,
        ].transpose(1, 2)
        inputs_v = inputs_v + inputs_pos_v
        # outputs shape: n_words, n_head, emb_dim
        outputs = torch.einsum("whl, whld->whd", inputs_attn, inputs_v)
        outputs = self.sdpa_bn_layer(outputs)
        return outputs

    def forward(self, char_code, qkv, max_len):
        # mask: n_words, 1, 1, max_len
        char_code = char_code[:, :max_len]
        not_mask = (
            (char_code != self.pad_idx).unsqueeze(-2).unsqueeze(-2).float()
        )
        # word_ft shape: n_words, emb_dim, n_head
        word_ft = self.scaled_dot_product_attention(
            char_code, not_mask, qkv, max_len
        ).transpose(-1, -2)
        # word_ft shape: n_words, emb_dim, n_head
        word_ft = self.mlp(word_ft)
        # word_ft shape: n_words, emb_dim
        word_ft = word_ft.reshape([word_ft.shape[0], -1])
        return word_ft

    def encode_token(self, char_code, char_len):
        qkv = self.compute_qkv()
        token_ft = torch.cat(
            [self(code, qkv, l[-1]) for code, l in zip(char_code, char_len)],
            dim=0,
        )
        return token_ft

    def eval_encode_token(self, char_code, char_len):
        qkv = self.compute_qkv()
        token_ft = torch.cat(
            [
                self(code, qkv, l[-1])
                for code, l in zip(
                    char_code.split(self.eval_sub_batch_size),
                    char_len.split(self.eval_sub_batch_size),
                )
            ],
            dim=0,
        )
        return token_ft
# SVT:1 ends here

# SVTransformerEncoder

# Transformer Encoder


# [[file:~/Works/char2token2mention/module/encoder.org::*SVTransformerEncoder][SVTransformerEncoder:1]]
class SVTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        n_head,
        emb_dim,
        dropout,
        pad_idx=0,
        device="cuda",
        eval_sub_batch_size=512
    ):
        super().__init__()
        # attributes
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.device = device
        self.pad_idx = pad_idx
        self.char_emb = nn.Embedding(
            vocab_size, self.emb_dim, padding_idx=pad_idx
        )
        # self.pos: 1, max_len, emb_dim
        self.pos = self.get_pos(self.max_len, self.emb_dim).to(device)
        self.qkv_layer = nn.Linear(emb_dim, 3 * emb_dim * n_head, bias=False)
        # sdpa: scaled dot product attention
        self.bn_layer0 = nn.LayerNorm(int(emb_dim), eps=1e-5)
        self.bn_layer1 = nn.LayerNorm(int(emb_dim), eps=1e-5)
        self.bn_layer2 = nn.LayerNorm(int(emb_dim), eps=1e-5)
        self.sdpa_temperature = emb_dim ** 0.5
        self.sdpa_dropout = nn.Dropout(dropout)
        self.sdpa_bn_layer = nn.LayerNorm(int(emb_dim), eps=1e-5)
        self.sdpa_fc = nn.Linear(emb_dim * n_head, emb_dim)
        # layers
        self.feedforward = FeedForward(emb_dim, emb_dim)
        self.eval_sub_batch_size = eval_sub_batch_size

    def get_pos(self, max_len, emb_dim):
        emb_dim_half = emb_dim >> 1
        freq = torch.pow(torch.tensor([1e4]), -1 / emb_dim_half).repeat(
            [emb_dim_half]
        )
        freq[0] = 1.0
        freq = freq.cumprod(-1)
        position = torch.arange(0, max_len)
        phase = torch.einsum("i, j->ij", position, freq)
        pos_embedding = torch.zeros([max_len, emb_dim])
        pos_embedding[:, 0::2] = torch.sin(phase)
        pos_embedding[:, 1::2] = torch.cos(phase)
        return pos_embedding.unsqueeze(0)

    def compute_qkv(self, inputs):
        new_shape = [inputs.size(0), inputs.size(1), self.n_head, -1]
        q, k, v = self.qkv_layer(inputs).reshape(new_shape).transpose(1, 2).chunk(3, dim=-1)
        return q, k, v

    def scaled_dot_product_attention(self, inputs, attn_mask, max_len):
        inputs = inputs + self.pos[:, :max_len]
        # q: batch_size, n_head, max_len, h_dim
        q, k, v = self.compute_qkv(inputs)
        qk = q @ k.transpose(-1, -2) / self.sdpa_temperature
        qk = qk.masked_fill(attn_mask, float("-inf"))
        # n_words, n_head, max_len, max_len
        attn_score = self.sdpa_dropout(
            nn.functional.softmax(qk, dim=-1)
        )
        # n_words, n_head, max_len
        not_mask = attn_mask.logical_not().float()
        not_mask_n_words = not_mask.sum([-1])
        attn_score = (not_mask @ attn_score).squeeze(-2) / not_mask_n_words
        # outputs shape: n_words, n_head, emb_dim
        outputs = torch.einsum("whl, whld->whd", attn_score, v)
        outputs = self.sdpa_fc(outputs.reshape([outputs.size(0), -1]))
        outputs = self.sdpa_bn_layer(outputs)
        return outputs

    def forward(self, char_code, max_len):
        # mask: n_words, 1, 1, max_len
        max_len = max_len.max()
        char_code = char_code[:, :max_len]
        inputs = self.bn_layer0(self.char_emb(char_code) + self.pos[:, :max_len])
        attn_mask = (
            (char_code == self.pad_idx).unsqueeze(-2).unsqueeze(-2)
        )
        # word_ft shape: n_words, emb_dim
        word_ft = self.scaled_dot_product_attention(
            inputs, attn_mask, max_len
        )
        word_ft = self.bn_layer1(word_ft + inputs.mean(1))
        word_ft = self.feedforward(word_ft) + word_ft
        word_ft = self.bn_layer2(word_ft)
        return word_ft
# SVTransformerEncoder:1 ends here

# SVTISDPA

# Index scaled dot production attention


# [[file:~/Works/char2token2mention/module/encoder.org::*SVTISDPA][SVTISDPA:1]]
class SVTISDPA(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        n_head,
        emb_dim,
        dropout,
        pad_idx=0,
        device="cuda",
        eval_sub_batch_size=512
    ):
        super().__init__()
        # attributes
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.device = device
        self.pad_idx = pad_idx
        self.mha_dim = n_head
        self.char_emb = nn.Embedding(
            vocab_size, self.emb_dim, padding_idx=pad_idx
        ).to(device)
        self.char = self.char_emb.weight
        self.pos = self.get_pos(self.max_len, emb_dim).to(device)
        self.qkv_layer = nn.Linear(emb_dim, 3 * emb_dim * n_head, bias=False)
        # sdpa: scaled dot product attention
        self.sdpa_temperature = emb_dim ** 0.5
        self.sdpa_dropout = nn.Dropout(dropout)
        self.sdpa_bn_layer = nn.LayerNorm(int(emb_dim), eps=1e-5)
        self.sdpa_fc = nn.Linear(emb_dim * n_head, emb_dim)
        # layers
        self.feedforward = FeedForward(emb_dim, emb_dim)
        self.bn_layer = nn.LayerNorm(int(emb_dim), eps=1e-5)
        self.eval_sub_batch_size = eval_sub_batch_size

    def get_pos(self, max_len, emb_dim):
        emb_dim_half = emb_dim >> 1
        freq = torch.pow(torch.tensor([1e4]), -1 / emb_dim_half).repeat(
            [emb_dim_half]
        )
        freq[0] = 1.0
        freq = freq.cumprod(-1)
        position = torch.arange(0, max_len)
        phase = torch.einsum("i, j->ij", position, freq)
        pos_embedding = torch.zeros([max_len, emb_dim])
        pos_embedding[:, 0::2] = torch.sin(phase)
        pos_embedding[:, 1::2] = torch.cos(phase)
        return pos_embedding

    def compute_qkv(self):
        # vocab_size, n_head, dim
        new_char_shape = [self.char.size(0), self.n_head, -1]
        char_q, char_k, char_v = self.qkv_layer(self.char).reshape(new_char_shape).chunk(3, dim=-1)
        # max_len, n_head, dim
        new_pos_shape = [self.pos.size(0), self.n_head, -1]
        pos_q, pos_k, pos_v = self.qkv_layer(self.pos).reshape(new_pos_shape).chunk(3, dim=-1)
        return self.interactive_char_pos(char_q, char_k, char_v, pos_q, pos_k, pos_v)

    def interactive_char_pos(self, char_q, char_k, char_v, pos_q, pos_k, pos_v):
        # (vocab_size, n_head, emb_dim) -> (n_head, vocab_size, emb_dim)
        char_q = char_q.transpose(0, 1)
        # (vocab_size, n_head, emb_dim) -> (n_head, emb_dim, vocab_size)
        char_k = char_k.permute(1, -1, 0)
        # n_head, vocab_size, vocab_size
        char_qk = (char_q @ char_k) / self.sdpa_temperature
        char_qk[:, :, self.pad_idx] = -1e9
        # (max_len, n_head, emb_dim) -> (n_head, max_len, emb_dim)
        pos_q = pos_q.transpose(0, 1)
        # (max_len, n_head, emb_dim) -> (n_head, emb_dim, max_len)
        pos_k = pos_k.permute(1, -1, 0)
        # (max_len, n_head, emb_dim) -> (1, n_head, max_len, emb_dim)
        pos_qk = (pos_q @ pos_k).unsqueeze(0) / self.sdpa_temperature
        # n_head, vocab_size, max_len
        char_pos_qk = (
            (char_q @ pos_k) + (pos_q @ char_k).transpose(-1, -2)
        ) / self.sdpa_temperature
        # max_len, n_head, emb_dim -> 1, n_head, max_len, emb_dim
        pos_v = pos_v.transpose(0, 1).unsqueeze(0)
        return char_qk, pos_qk, char_pos_qk, char_v, pos_v

    def scaled_dot_product_attention(self, inputs, not_mask, qkv, max_len):
        char_qk, pos_qk, char_pos_qk, char_v, pos_v = qkv
        # (n_head, vocab_size, vocab_size) -> (n_head, n_words, max_len, vocab_size)
        inputs_char_qk = char_qk[
            :,
            inputs,
        ]
        column_indexes = torch.einsum(
            "ij, ik -> ijk", torch.ones_like(inputs), inputs
        ).repeat([self.n_head, 1, 1, 1])
        # (n_head, n_words, max_len, vocab_size) -> (n_head, n_words, max_len, max_len)
        inputs_char_qk = inputs_char_qk.gather(-1, column_indexes)
        # (n_head, n_words, max_len, max_len) -> (n_words, n_head, max_len, max_len)
        inputs_char_qk = inputs_char_qk.transpose(0, 1)
        # (n_head, vocab_size, max_len) -> (n_words, n_head, max_len, max_len)
        inputs_char_pos_qk = char_pos_qk[:, inputs, :max_len].transpose(0, 1)
        # n_words, n_head, max_len, max_len
        inputs_sim = (
            inputs_char_qk
            + inputs_char_pos_qk
            + pos_qk[:, :, :max_len, :max_len]
        )
        # n_words, n_head, max_len, max_len
        inputs_attn = self.sdpa_dropout(
            nn.functional.softmax(inputs_sim, dim=-1)
        )
        # n_words, n_head, max_len
        not_mask_n_words = not_mask.sum(-1)
        inputs_attn = (not_mask @ inputs_attn).squeeze(-2) / not_mask_n_words
        # 1, n_head, max_len, emb_dim
        inputs_pos_v = pos_v[:, :, :max_len]
        # max_len, n_head, emb_dim -> n_words, n_head, max_len, emb_dim
        inputs_v = char_v[
            inputs,
        ].transpose(1, 2)
        inputs_v = inputs_v + inputs_pos_v
        # outputs shape: n_words, n_head, emb_dim
        outputs = torch.einsum("whl, whld->whd", inputs_attn, inputs_v)
        outputs = self.sdpa_fc(outputs.reshape([outputs.size(0), -1]))
        outputs = self.sdpa_bn_layer(outputs)
        return outputs

    def forward(self, char_code, qkv, max_len):
        # mask: n_words, 1, 1, max_len
        char_code = char_code[:, :max_len]
        not_mask = (
            (char_code != self.pad_idx).unsqueeze(-2).unsqueeze(-2).float()
        )
        # word_ft shape: n_words, emb_dim * n_head
        word_ft = self.scaled_dot_product_attention(
            char_code, not_mask, qkv, max_len
        ).reshape([char_code.size(0), -1])
        # word_ft shape: n_words, emb_dim
        word_ft = self.feedforward(word_ft) + word_ft
        word_ft = self.bn_layer(word_ft)
        return word_ft

    def encode_token(self, char_code, char_len):
        qkv = self.compute_qkv()
        token_ft = torch.cat(
            [self(code, qkv, l[-1]) for code, l in zip(char_code, char_len)],
            dim=0,
        )
        return token_ft

    def eval_encode_token(self, char_code, char_len):
        qkv = self.compute_qkv()
        token_ft = torch.cat(
            [
                self(code, qkv, l[-1])
                for code, l in zip(
                    char_code.split(self.eval_sub_batch_size),
                    char_len.split(self.eval_sub_batch_size),
                )
            ],
            dim=0,
        )
        return token_ft
# SVTISDPA:1 ends here

# SVTMLP

# [[file:~/Works/char2token2mention/module/encoder.org::*SVTMLP][SVTMLP:1]]
class SVTMLP(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        n_head,
        emb_dim,
        dropout,
        mlp_config,
        pad_idx=0,
        device="cuda",
        eval_sub_batch_size=512
    ):
        super().__init__()
        # attributes
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.mlp_config = mlp_config
        self.device = device
        self.pad_idx = pad_idx
        self.mha_dim = n_head
        self.char_emb = nn.Embedding(
            vocab_size, self.emb_dim, padding_idx=pad_idx
        ).to(device)
        self.char = self.char_emb.weight
        self.pos = self.get_pos(self.max_len, self.emb_dim).to(device)
        self.qkv_layer = ComputeQKV(self.n_head, self.emb_dim)
        # sdpa: scaled dot product attention
        self.sdpa_temperature = emb_dim ** 0.5
        self.sdpa_dropout = nn.Dropout(dropout)
        self.sdpa_bn_layer = nn.LayerNorm(emb_dim, eps=1e-5)
        # layers
        self.mlp_dims = [self.mha_dim] + [c[0] for c in mlp_config[:-1]]
        self.mlp = nn.Sequential(
            *[Dense(i, *c) for i, c in zip(self.mlp_dims, mlp_config)]
        )
        self.eval_sub_batch_size = eval_sub_batch_size

    def get_pos(self, max_len, emb_dim):
        emb_dim_half = emb_dim >> 1
        freq = torch.pow(torch.tensor([1e4]), -1 / emb_dim_half).repeat(
            [emb_dim_half]
        )
        freq[0] = 1.0
        freq = freq.cumprod(-1)
        position = torch.arange(0, max_len)
        phase = torch.einsum("i, j->ij", position, freq)
        pos_embedding = torch.zeros([max_len, emb_dim])
        pos_embedding[:, 0::2] = torch.sin(phase)
        pos_embedding[:, 1::2] = torch.cos(phase)
        return pos_embedding

    def compute_qkv(self):
        char_qkv = self.qkv_layer(self.char)
        pos_qkv = self.qkv_layer(self.pos)
        return self.interactive_char_pos([char_qkv, pos_qkv])

    def interactive_char_pos(self, qkv):
        char_qkv, pos_qkv = qkv
        char_q, char_k, char_v = [
            char_qkv[:, i] for i in range(char_qkv.shape[1])
        ]
        pos_q, pos_k, pos_v = [pos_qkv[:, i] for i in range(pos_qkv.shape[1])]
        # (vocab_size, n_head, emb_dim) -> (n_head, vocab_size, emb_dim)
        char_q = char_q.transpose(0, 1)
        # (vocab_size, n_head, emb_dim) -> (n_head, emb_dim, vocab_size)
        char_k = char_k.permute(1, -1, 0)
        # n_head, vocab_size, vocab_size
        char_qk = (char_q @ char_k) / self.sdpa_temperature
        char_qk[:, :, self.pad_idx] = -1e9
        # (max_len, n_head, emb_dim) -> (n_head, max_len, emb_dim)
        pos_q = pos_q.transpose(0, 1)
        # (max_len, n_head, emb_dim) -> (n_head, emb_dim, max_len)
        pos_k = pos_k.permute(1, -1, 0)
        # (max_len, n_head, emb_dim) -> (1, n_head, max_len, emb_dim)
        pos_qk = (pos_q @ pos_k).unsqueeze(0) / self.sdpa_temperature
        # n_head, vocab_size, max_len
        char_pos_qk = (
            (char_q @ pos_k) + (pos_q @ char_k).transpose(-1, -2)
        ) / self.sdpa_temperature
        # max_len, n_head, emb_dim -> 1, n_head, max_len, emb_dim
        pos_v = pos_v.transpose(0, 1).unsqueeze(0)
        return char_qk, pos_qk, char_pos_qk, char_v, pos_v

    def scaled_dot_product_attention(self, inputs, not_mask, qkv, max_len):
        char_qk, pos_qk, char_pos_qk, char_v, pos_v = qkv
        # (n_head, vocab_size, vocab_size) -> (n_head, n_words, max_len, vocab_size)
        inputs_char_qk = char_qk[
            :,
            inputs,
        ]
        column_indexes = torch.einsum(
            "ij, ik -> ijk", torch.ones_like(inputs), inputs
        ).repeat([self.n_head, 1, 1, 1])
        # (n_head, n_words, max_len, vocab_size) -> (n_head, n_words, max_len, max_len)
        inputs_char_qk = inputs_char_qk.gather(-1, column_indexes)
        # (n_head, n_words, max_len, max_len) -> (n_words, n_head, max_len, max_len)
        inputs_char_qk = inputs_char_qk.transpose(0, 1)
        # (n_head, vocab_size, max_len) -> (n_words, n_head, max_len, max_len)
        inputs_char_pos_qk = char_pos_qk[:, inputs, :max_len].transpose(0, 1)
        # n_words, n_head, max_len, max_len
        inputs_sim = (
            inputs_char_qk
            + inputs_char_pos_qk
            + pos_qk[:, :, :max_len, :max_len]
        )
        # n_words, n_head, max_len, max_len
        inputs_attn = self.sdpa_dropout(
            nn.functional.softmax(inputs_sim, dim=-1)
        )
        # n_words, n_head, max_len
        not_mask_n_words = not_mask.sum(-1)
        inputs_attn = (not_mask @ inputs_attn).squeeze(-2) / not_mask_n_words
        # 1, n_head, max_len, emb_dim
        inputs_pos_v = pos_v[:, :, :max_len]
        # max_len, n_head, emb_dim -> n_words, n_head, max_len, emb_dim
        inputs_v = char_v[
            inputs,
        ].transpose(1, 2)
        inputs_v = inputs_v + inputs_pos_v
        # outputs shape: n_words, n_head, emb_dim
        outputs = torch.einsum("whl, whld->whd", inputs_attn, inputs_v)
        outputs = self.sdpa_bn_layer(outputs)
        return outputs

    def forward(self, char_code, qkv, max_len):
        # mask: n_words, 1, 1, max_len
        char_code = char_code[:, :max_len]
        not_mask = (
            (char_code != self.pad_idx).unsqueeze(-2).unsqueeze(-2).float()
        )
        # word_ft shape: n_words, emb_dim, n_head
        word_ft = self.scaled_dot_product_attention(
            char_code, not_mask, qkv, max_len
        ).transpose(-1, -2)
        # word_ft shape: n_words, emb_dim, n_head
        word_ft = self.mlp(word_ft)
        # word_ft shape: n_words, emb_dim
        word_ft = word_ft.reshape([word_ft.shape[0], -1])
        return word_ft

    def encode_token(self, char_code, char_len):
        qkv = self.compute_qkv()
        token_ft = torch.cat(
            [self(code, qkv, l[-1]) for code, l in zip(char_code, char_len)],
            dim=0,
        )
        return token_ft

    def eval_encode_token(self, char_code, char_len):
        qkv = self.compute_qkv()
        token_ft = torch.cat(
            [
                self(code, qkv, l[-1])
                for code, l in zip(
                    char_code.split(self.eval_sub_batch_size),
                    char_len.split(self.eval_sub_batch_size),
                )
            ],
            dim=0,
        )
        return token_ft
# SVTMLP:1 ends here

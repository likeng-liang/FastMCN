import torch
from torch import nn

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

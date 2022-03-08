# Import


# [[file:~/Works/char2token2mention/model/layer.org::*Import][Import:1]]
import torch
from torch import nn
from torch.nn import Module

# Import:1 ends here

# Call


# [[file:~/Works/char2token2mention/model/layer.org::*Call][Call:1]]
def call(inputs, func):
    return func(inputs)


# Call:1 ends here

# Identity


# [[file:~/Works/char2token2mention/model/layer.org::*Identity][Identity:1]]
def identity(inputs):
    return inputs


# Identity:1 ends here

# Dense


# [[file:~/Works/char2token2mention/model/layer.org::*Dense][Dense:1]]
class Dense(Module):
    def __init__(self, input_dim, output_dim, hook, bias=True):
        super().__init__()
        self.__dict__.update(locals())
        args_names = [key for key in args if key != "self"]
        for n in args_names:
            setattr(self, n, args[n])
        # self.act_func = torch.jit.script(getattr(activate, activate_func)())
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, inputs):
        ft = self.linear(inputs)
        result = reduce(call, self.hook, ft)
        return result


# Dense:1 ends here

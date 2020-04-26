from collections import OrderedDict

from torch import nn

from ..modules.utils import append_params


class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        self.params = OrderedDict()
        self.layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Sequential(
                #nn.Dropout(0.5),
                nn.Linear(512 * 3 * 3, 256),
                nn.ReLU())),
            ('fc2', nn.Sequential(
                #nn.Dropout(0.5),
                nn.Linear(256, 1 * 3 * 3)))]))
        self.build_param_dict()

    def build_param_dict(self):
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def forward(self, x):
        #
        # forward model
        for _, module in self.layers.named_children():
            x = module(x)

        return x

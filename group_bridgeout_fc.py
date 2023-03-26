import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
from torch.autograd import Variable

dev_str = 'cuda' if torch.cuda.is_available() else 'cpu'
rand_gen = torch.Generator(device=dev_str)


class GroupBridgeoutFcLayer(Module):
    r"""TODO
    """
    __constants__ = ['in_features', 'out_features']

    def __init__(
            self,
            in_features,
            out_features,
            nu=0.5,
            device=None,
            dtype=None,
            bias=True):
        super(GroupBridgeoutFcLayer, self).__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        # TODO
        self.nu = nu

        self.p = 0.5
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_x):
        if self.training and self.nu > 0.0:
            bppt, batch_size, feature_size = input_x.size()
            regularization_strength = self.nu / batch_size

            w = self.weight
            # l2 raised to the power 0.5 so that the effective penalty is L1 of L2 norms.
            wq = torch.norm(w, 2, dim=1).mul(regularization_strength).pow(0.5).unsqueeze(1)

            # take input, clone to get same dim mask
            noise = w.data.clone()
            # generate: (bernoulli / (1-p)) - 1
            # if p=0.5, then a matrix of -1 and 1
            noise.bernoulli_(1 - self.p, generator=rand_gen).div_(1 - self.p).sub_(1)

            # Targeting
            # #--------------------------------------------------------
            # # targetting
            # target_fraction = 0.75
            # n_features_to_drop = int(len(wq)*target_fraction)
            # sorted_indices = torch.argsort(wq, dim=0)[:,0]
            # nth_ranked_value = wq[sorted_indices[n_features_to_drop]]
            # targeting_mask = wq.lt(nth_ranked_value).type(wq.dtype).to(wq.device)
            # #--------------------------------------------------------
            # Apparently, targeting reduces sparsity, so we do not perform targeting
            targeting_mask = 1.0

            perturbation_equivalent = wq.mul(Variable(noise)).mul(targeting_mask)
            w = w.add(perturbation_equivalent)
            output = F.linear(input_x, w, self.bias)
        else:
            output = F.linear(input_x, self.weight, self.bias)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

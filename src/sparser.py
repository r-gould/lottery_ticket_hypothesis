import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn.utils import prune

class Sparser:
    """
    Responsible for producing a network of greater sparsity.
    """

    def __init__(self, prune_rates, num_prunes):
        for key, rate in prune_rates.items():
            prune_rates[key] = ((100 * rate) ** (1/num_prunes)) / 100

        self.prune_rates = prune_rates
        self.num_prunes = num_prunes

    def init_prunes(self, net):
        for module in net.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.identity(module, "weight")

        self.init_states = deepcopy(net.state_dict())

    def reset(self, net):
        conv_rate, fc_rate = self.prune_rates["conv"], self.prune_rates["fc"]

        with torch.no_grad():
            for name, module in net.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    rate = conv_rate if isinstance(module, nn.Conv2d) else fc_rate
                    prune.l1_unstructured(module, name="weight", amount=rate)
                    
                    init_weight = self.init_states[name + ".weight_orig"]
                    init_bias = self.init_states[name + ".bias"]
                    module.weight_orig.copy_(init_weight)
                    module.bias.copy_(init_bias)
                    module.weight = init_weight * module.weight_mask
        
        return net
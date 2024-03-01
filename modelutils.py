# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
import torch.nn as nn
from attention import FusedQK


def find_layers(module, layers=[nn.Conv2d, nn.Linear, FusedQK], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

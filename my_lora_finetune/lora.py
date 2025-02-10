import torch
from torch import nn
import math
from config import *

class LoraLayer(nn.Module):
    def __init__(self, raw_linear, in_features, out_features, r, alpha, device):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.lora_a = nn.Parameter(torch.empty(in_features, r, device=device, dtype=torch.bfloat16))
        self.lora_b = nn.Parameter(torch.zeros(r, out_features, device=device, dtype=torch.bfloat16))

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.raw_linear = raw_linear        # 这里是获取原模型的Linear层

    def forward(self, x):
        raw_output = self.raw_linear(x)
        lora_output = x @ ((self.lora_a @ self.lora_b) * self.alpha/self.r)
        return raw_output + lora_output           # 返回的结果是lora层+原始层，但是原始层被冻结，所以训练的是dota W

def inject_lora(model, name, layer, device):
    name_cols = name.split('.')

    # 逐层往下找，按name找到当前linear上一层的module
    children = name_cols[:-1]  # 找到倒数第二层
    cur_module_layer = model
    for child in children:
        cur_module_layer = getattr(cur_module_layer, child)

    # print(layer == cur_module_layer.getattr(name_cols[-1]))   # cur_module_layer是layer的上一层
    lora_layer = LoraLayer(layer, layer.in_features, layer.out_features, LORA_R, LORA_ALPHA, device=device)
    # 最后将lora注入到原始模型中，通过setattr修改cur_module_layer的子层
    setattr(cur_module_layer, name_cols[-1], lora_layer)



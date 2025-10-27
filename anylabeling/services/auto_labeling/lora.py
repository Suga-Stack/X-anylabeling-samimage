import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class LoRALayer(nn.Module):
    """LoRA base layer"""
    def __init__(self, in_features: int, out_features: int, r: int, lora_alpha: float = 1.0):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)


class Linear(nn.Linear, LoRALayer):
    """LoRA linear layer"""
    def __init__(self, in_features: int, out_features: int, r: int = 0, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, in_features, out_features, r)
        self.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            result += (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result


class MergedLinear(nn.Linear, LoRALayer):
    """Merged LoRA linear layer (for QKV)"""
    def __init__(self, in_features: int, out_features: int, r: int = 0, enable_lora: List[bool] = None, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        
        if enable_lora is None:
            enable_lora = [True, True, True]
            
        self.enable_lora = enable_lora
        self.r = r
        
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(torch.zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r * sum(enable_lora))))
            self.scaling = 1.0 / r
            self.reset_parameters()
            self.weight.requires_grad = False

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
            result += lora_output
        return result


def prepare_lora(model_type, model: nn.Module, r: int):
    """Prepare LoRA layers - adapt this to match your training setup"""
    from .sam_modeling import Attention, EncoderAttention
    
    for name, module in model.named_children():
        if 'neck' in name:
            continue
        if isinstance(module, Attention):
            q_proj = module.q_proj
            v_proj = module.v_proj
            new_q_proj = Linear(q_proj.in_features, q_proj.out_features, r=r)
            new_v_proj = Linear(v_proj.in_features, v_proj.out_features, r=r)
            setattr(module, 'q_proj', new_q_proj)
            setattr(module, 'v_proj', new_v_proj)
        elif isinstance(module, EncoderAttention):
            qkv = module.qkv
            setattr(module, 'qkv', MergedLinear(qkv.in_features, qkv.out_features, r, enable_lora=[True, False, True]))
        elif ('rep' in model_type) and isinstance(module, nn.Conv2d) and module.kernel_size[0] == 1 and module.groups==1:
            # Add ConvLoRA if needed
            pass
        else:
            prepare_lora(model_type, module, r)


def mark_only_lora_as_trainable(model: nn.Module):
    """Only mark LoRA parameters as trainable"""
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
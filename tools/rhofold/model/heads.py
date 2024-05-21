# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from inspect import isfunction

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = self.a_2*(x-mean)
        x /= std
        x += self.b_2

        return x

class FeedForwardLayer(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff,
                 p_drop = 0.1,
                 d_model_out = None,
                 is_post_act_ln = False,
                 **unused,
                 ):

        super(FeedForwardLayer, self).__init__()
        d_model_out = default(d_model_out, d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.post_act_ln = LayerNorm(d_ff) if is_post_act_ln else nn.Identity()
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_ff, d_model_out)
        self.activation = nn.ReLU()

    def forward(self, src):
        src = self.linear2(self.dropout(self.post_act_ln(self.activation(self.linear1(src)))))
        return src


class DistHead(nn.Module):
    def __init__(self,
                 c_in,
                 no_bins=40,
                 **kwargs):
        super(DistHead, self).__init__()
        self.norm = LayerNorm(c_in)
        self.proj = nn.Linear(c_in, c_in)

        self.resnet_dist_0 = FeedForwardLayer(d_model=c_in, d_ff=c_in * 4, d_model_out=no_bins,
                                            **kwargs)
        self.resnet_dist_1 = FeedForwardLayer(d_model=c_in, d_ff=c_in * 4, d_model_out=no_bins,
                                            **kwargs)
        self.resnet_dist_2 = FeedForwardLayer(d_model=c_in, d_ff=c_in * 4, d_model_out=no_bins,
                                            **kwargs)

    def forward(self, x):

        x = self.norm(x)
        x = self.proj(x)

        logits_dist0 = self.resnet_dist_0(x).permute(0, 3, 1, 2)
        logits_dist1 = self.resnet_dist_1(x).permute(0, 3, 1, 2)
        logits_dist2 = self.resnet_dist_2(x).permute(0, 3, 1, 2)

        return logits_dist0, logits_dist1, logits_dist2

class SSHead(nn.Module):
    def __init__(self,
                 c_in,
                 no_bins=1,
                 **kwargs):
        super(SSHead, self).__init__()
        self.norm = LayerNorm(c_in)
        self.proj = nn.Linear(c_in, c_in)
        self.ffn = FeedForwardLayer(d_model=c_in, d_ff = c_in*4, d_model_out=no_bins, **kwargs)

    def forward(self, x):

        x = self.norm(x)
        x = self.proj(x)
        x = 0.5 * (x + x.permute(0, 2, 1, 3))
        logits = self.ffn(x).permute(0, 3, 1, 2)

        return logits

class pLDDTHead(nn.Module):
    def __init__(self, c_in, no_bins = 50):
        super(pLDDTHead, self).__init__()

        self.bin_vals = (torch.arange(no_bins).view(1, 1, -1) + 0.5) / no_bins

        self.net_lddt = nn.Sequential(
            nn.LayerNorm(c_in),
            nn.Linear(c_in, c_in),
            nn.ReLU(),
            nn.Linear(c_in, c_in),
            nn.ReLU(),
            nn.Linear(c_in, no_bins),
        )
        self.sfmx = nn.Softmax(dim=2)

    def forward(self, sfea_tns):

        logits = self.net_lddt(sfea_tns)

        self.bin_vals = self.bin_vals.to(logits.device)

        plddt_local = torch.sum(self.bin_vals * self.sfmx(logits), dim=2)

        plddt_global = torch.mean(plddt_local, dim=1)

        return  plddt_local,  plddt_global
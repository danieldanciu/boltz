"""Utility functions for initializing weights and biases."""

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
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

import math

import numpy as np
import torch
from scipy.stats import truncnorm


def _calculate_fan(linear_weight_shape: torch.Size, fan: str = "fan_in") -> int:
    fan_out, fan_in = linear_weight_shape
    if fan == "fan_in":
        return fan_in
    if fan == "fan_out":
        return fan_out
    if fan == "fan_avg":
        return (fan_in + fan_out) / 2
    raise ValueError("Invalid fan option")


def trunc_normal_init_(weights: torch.Tensor, scale: float = 1.0, fan: str = "fan_in"):
    """Initialize weights using a truncated normal distribution.

    Args:
        weights (torch.Tensor): The weights to initialize.
        scale (float): The scale of the distribution.
        fan (str): The fan mode to use ('fan_in', 'fan_out', or 'fan_avg').
    """
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = math.prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights: torch.Tensor):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights: torch.Tensor):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights: torch.Tensor):
    torch.nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights: torch.Tensor):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights: torch.Tensor):
    with torch.no_grad():
        weights.fill_(0.0)


def bias_init_zero_(bias: torch.Tensor):
    with torch.no_grad():
        bias.fill_(0.0)


def bias_init_one_(bias: torch.Tensor):
    with torch.no_grad():
        bias.fill_(1.0)


def normal_init_(weights: torch.Tensor):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")

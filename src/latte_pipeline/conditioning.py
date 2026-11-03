"""Interchangeable numerical conditioning prefixes, independent of field tokens."""
import torch
from torch import nn


class ParameterEncoder(nn.Module):
    def __init__(self, statistics, width, encoding='binned', bins=64):
        super().__init__()
        if encoding not in ('binned', 'continuous') or bins < 2 or not statistics:
            raise ValueError('Invalid parameter encoder configuration')
        self.encoding, self.bins = encoding, bins
        self.register_buffer('minimum', torch.tensor([row['minimum'] for row in statistics], dtype=torch.float64))
        self.register_buffer('maximum', torch.tensor([row['maximum'] for row in statistics], dtype=torch.float64))
        self.register_buffer('mean', torch.tensor([row['mean'] for row in statistics], dtype=torch.float64))
        self.register_buffer('scale', torch.tensor([row['scale'] for row in statistics], dtype=torch.float64))
        self.identities = nn.Embedding(len(statistics), width)
        if encoding == 'binned':
            self.values = nn.Embedding(len(statistics) * (bins + 3), width)
        else:
            self.values = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))

    def token_indices(self, values):
        if values.ndim != 2 or values.shape[1] != len(self.minimum):
            raise ValueError('Parameter order/shape mismatch')
        values = values.double()
        present = torch.isfinite(values)
        fraction = (torch.nan_to_num(values) - self.minimum) / (self.maximum - self.minimum).clamp_min(1e-30)
        indices = 1 + torch.round(fraction.clamp(0, 1) * (self.bins - 1)).long()
        indices = torch.where(values < self.minimum, self.bins + 1, indices)
        indices = torch.where(values > self.maximum, self.bins + 2, indices)
        indices = torch.where(present, indices, 0)
        return indices + torch.arange(len(self.minimum), device=values.device) * (self.bins + 3)

    def forward(self, values):
        if values.ndim != 2 or values.shape[1] != len(self.minimum):
            raise ValueError('Parameter order/shape mismatch')
        identities = self.identities(torch.arange(values.shape[1], device=values.device))[None]
        if self.encoding == 'binned':
            encoded = self.values(self.token_indices(values))
        else:
            present = torch.isfinite(values)
            standardized = torch.where(present, (values.double() - self.mean) / self.scale, 0.).float()
            encoded = self.values(torch.stack((standardized, present.float()), -1))
        return identities + encoded


class ConditioningPrefix(nn.Module):
    def __init__(self, statistics, width, time_scale, encoding='binned', bins=64):
        super().__init__()
        if time_scale <= 0:
            raise ValueError('Time scale must be positive')
        self.parameters_encoder = None if encoding == 'none' else ParameterEncoder(statistics, width, encoding, bins)
        self.register_buffer('time_scale', torch.tensor(float(time_scale)))
        self.time = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.dataset = nn.Parameter(torch.zeros(1, 1, width))

    def forward(self, parameters, source_time, target_time):
        times = torch.stack((source_time, target_time, target_time - source_time), -1).float() / self.time_scale
        prefix = [self.dataset.expand(len(parameters), -1, -1), self.time(times)[:, None]]
        if self.parameters_encoder is not None:
            prefix.append(self.parameters_encoder(parameters))
        return torch.cat(prefix, dim=1)

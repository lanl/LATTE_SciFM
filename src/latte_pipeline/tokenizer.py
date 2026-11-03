"""Frozen Phaedra encoding and physical inversion with explicit provenance."""
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_ema import ExponentialMovingAverage

from .data import normalize, physical

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'third_party' / 'Phaedra'))


def make_tokenizer(config):
    from phaedra.phaedra_model import PhaedraModel
    return PhaedraModel(OmegaConf.create(config).tokenizer_hyperparameters)


def load_tokenizer(path, device, expected_step=None):
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if expected_step is not None and saved['step'] != expected_step:
        raise ValueError('Wrong tokenizer checkpoint step')
    model = make_tokenizer(saved['config'])
    model.load_state_dict(saved['model'], strict=True)
    ema = ExponentialMovingAverage(model.parameters(), decay=.999)
    ema.load_state_dict(saved['ema'])
    ema.copy_to(model.parameters())
    model.to(device).eval().requires_grad_(False)
    return model


def standardized_frame(frame, fields, normalization):
    return np.stack([normalize(values, normalization[field['name']]) for field, values in zip(fields, frame)])


@torch.inference_mode()
def encode_frame(model, frame, device):
    chunks = []
    for field in frame:
        tensor = torch.from_numpy(field[None, None]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            _, _, codes, _ = model.encode(tensor)
        values = np.stack([code[0].long().cpu().numpy() for code in codes], axis=0)
        if values.shape != (2, field.shape[0] // 8, field.shape[1] // 8):
            raise ValueError('Unexpected tokenizer geometry')
        if (values < 0).any() or values[0].max() >= 8640 or values[1].max() >= 1024:
            raise ValueError('Unexpected FSQ token alphabet')
        chunks.append(values.astype(np.uint16))
    return np.stack(chunks)


@torch.inference_mode()
def decode_frame(model, codes, fields, normalization, device, clip=True):
    predictions = []
    for values, field in zip(codes, fields):
        morphology, amplitude = [torch.as_tensor(array.astype(np.int64), device=device)[None] for array in values]
        latent = torch.cat((model.quantizer.indices_to_codes(morphology),
                            model.approximate_continuous.quantizer.indices_to_codes(amplitude)), 1)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            reconstructed = model.decode(latent).float()[0, 0].cpu().numpy()
        predictions.append(physical(reconstructed, normalization[field['name']], clip=clip))
    return np.stack(predictions)

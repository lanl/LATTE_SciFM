"""Evaluation-only incremental decoding for the shared 2D-RoPE LLaMA."""
import math

import torch
import torch.nn.functional as functional

from scripts.train_encdec_vq_transformer_pdegym6_llama import apply_2d_rope


def spatial_indices(source):
    batch, channels, height, width = source.shape
    rows, cols, fields = torch.meshgrid(torch.arange(height, device=source.device),
                                      torch.arange(width, device=source.device),
                                      torch.arange(channels, device=source.device), indexing="ij")
    return tuple(values.reshape(1, -1).expand(batch, -1) for values in (rows, cols, fields))


def heads(values, attention):
    return values.reshape(values.shape[0], values.shape[1], attention.n_head, attention.head_dim).transpose(1, 2)


def attend(attention, query, key, value, mask=None):
    if attention.use_sdpa:
        result = functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    else:
        scores = query @ key.transpose(-2, -1) / math.sqrt(attention.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        result = scores.softmax(dim=-1) @ value
    result = result.transpose(1, 2).reshape(query.shape[0], 1, -1)
    return attention.resid_drop(attention.proj(result))


class CachedDecoder:
    def __init__(self, model, memory, rows, cols, fields, mask_padding=False):
        if model.training:
            raise ValueError("Cached decoding requires eval mode")
        self.model = model
        self.rows, self.cols, self.fields = rows, cols, fields
        self.memory_mask = ~model.memory_padding_mask[:, None, None, :]
        self.layers = []
        self.position = 0
        self.keep = torch.ones_like(rows, dtype=torch.bool) if mask_padding else None
        for block in model.dec_blocks:
            attention = block.cross_attn
            key, value = attention.kv(memory).chunk(2, dim=-1)
            key, value = heads(key, attention), heads(value, attention)
            key, _ = apply_2d_rope(key, key, model.memory_rows, model.memory_cols)
            shape = (memory.shape[0], block.self_attn.n_head, rows.shape[1], block.self_attn.head_dim)
            self.layers.append((key, value, torch.empty(shape, device=key.device, dtype=key.dtype),
                                torch.empty(shape, device=key.device, dtype=key.dtype)))

    def step(self, token):
        position = self.position
        if position >= self.rows.shape[1] or token.shape != (self.rows.shape[0], 1):
            raise ValueError("Invalid incremental decoding position or token shape")
        rows, cols, fields = (values[:, position:position + 1] for values in (self.rows, self.cols, self.fields))
        model = self.model
        if self.keep is not None:
            self.keep[:, position:position + 1] = token.ne(model.pad_id)
        mask = None if self.keep is None else self.keep[:, None, None, :position + 1]
        state = model.drop(model.tok_emb(token) + model.row_emb(rows) + model.col_emb(cols) + model.channel_emb(fields))
        for block, (memory_key, memory_value, keys, values) in zip(model.dec_blocks, self.layers):
            attention = block.self_attn
            query, key, value = (heads(part, attention) for part in attention.qkv(block.ln1(state)).chunk(3, dim=-1))
            query, key = apply_2d_rope(query, key, rows, cols)
            keys[:, :, position:position + 1] = key
            values[:, :, position:position + 1] = value
            state = state + attend(attention, query, keys[:, :, :position + 1], values[:, :, :position + 1], mask)
            attention = block.cross_attn
            query = heads(attention.q(block.ln2(state)), attention)
            query, _ = apply_2d_rope(query, query, rows, cols)
            state = state + attend(attention, query, memory_key, memory_value, self.memory_mask)
            state = state + block.mlp(block.ln3(state))
        self.position += 1
        return model.lm_head(model.dec_ln(state))[:, 0]


@torch.inference_mode()
def predict_cached(model, source, horizons, bos_id, dataset_token):
    rows, cols, fields = spatial_indices(source)
    encoded = source.permute(0, 2, 3, 1).reshape(source.shape[0], -1)
    deck = torch.stack((torch.full_like(horizons, dataset_token), model.n_embed + horizons), dim=1)
    memory = model.encode(encoded, rows, cols, fields, deck_tokens=deck)
    decoder = CachedDecoder(model, memory, rows, cols, fields)
    token = torch.full((source.shape[0], 1), bos_id, device=source.device, dtype=torch.long)
    predicted = []
    for position in range(encoded.shape[1]):
        token = decoder.step(token)[:, :model.n_embed].argmax(dim=-1, keepdim=True)
        predicted.append(token)
    return torch.cat(predicted, dim=1).reshape(source.shape[0], model.Hq, model.Wq, model.C).permute(0, 3, 1, 2)


@torch.inference_mode()
def check_cached_logits(model, source, bos_id, dataset_token, steps=32):
    rows, cols, fields = spatial_indices(source)
    encoded = source.permute(0, 2, 3, 1).reshape(source.shape[0], -1)
    deck = torch.tensor([[dataset_token, model.n_embed + 20]], device=source.device).expand(source.shape[0], -1)
    memory = model.encode(encoded, rows, cols, fields, deck_tokens=deck)
    decoder = CachedDecoder(model, memory, rows, cols, fields)
    prefix = torch.full((source.shape[0], 1), bos_id, device=source.device, dtype=torch.long)
    maximum = 0.0
    for position in range(min(steps, encoded.shape[1])):
        cached = decoder.step(prefix[:, -1:])
        reference = model.decode(prefix, rows, cols, fields, memory)[:, -1]
        torch.testing.assert_close(cached, reference, rtol=2e-4, atol=3e-4)
        maximum = max(maximum, float((cached - reference).abs().max()))
        prefix = torch.cat((prefix, reference[:, :model.n_embed].argmax(dim=-1, keepdim=True)), dim=1)
    return maximum

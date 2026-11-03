#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import math
import time
import random
from typing import List
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from src.data.resumable import ResumableDistributedBatchSampler



class DistributedEvalSampler(Sampler):
    """Shard validation without DistributedSampler's duplicate padding."""
    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        n = len(self.dataset)
        if self.rank >= n:
            return 0
        return (n - 1 - self.rank) // self.world_size + 1

def ddp_is_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_setup():
    if not ddp_is_enabled():
        return 0, 1, 0, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda_device = int(os.environ.get("CUDA_DEVICE_ORDINAL", str(local_rank)))
    torch.cuda.set_device(cuda_device)
    return rank, world_size, local_rank, cuda_device


def ddp_cleanup():
    if ddp_is_enabled() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scaler(enabled: bool):
    return torch.amp.GradScaler('cuda', enabled=enabled)


def autocast_ctx(enabled: bool):
    return torch.amp.autocast('cuda', enabled=enabled)


class SharedFixedTokenDataset(Dataset):
    def __init__(self, token_files: List[str], max_pairs: int = None, seed: int = 1337, delta_t_choices: List[int] | None = None, num_dataset_tokens: int | None = None, dataset_slots: List[int] | None = None, channel_policy: str = 'all'):
        self.token_files = [str(x) for x in token_files]
        self.seed = int(seed)
        self.delta_t_choices = sorted({int(x) for x in (delta_t_choices or [1])})
        self.channel_policy = str(channel_policy)
        self.data_stream_id = (tuple(self.token_files), self.channel_policy, None if dataset_slots is None else tuple(dataset_slots))
        if self.channel_policy not in ('all', 'poseidon_matched_4_2'):
            raise RuntimeError(f'Unsupported channel_policy={self.channel_policy!r}')
        if len(self.delta_t_choices) == 0:
            raise RuntimeError("delta_t_choices must be non-empty")
        if min(self.delta_t_choices) <= 0:
            raise RuntimeError(f"delta_t_choices must be positive, got {self.delta_t_choices}")

        self.arrays = []
        self.metas = []
        n_embed_ref = None
        Hq_ref = None
        Wq_ref = None
        C_ref = None
        C_max = None
        pairs = []
        file_specs = []

        for file_i, token_file in enumerate(self.token_files):
            z = np.load(token_file, allow_pickle=True)
            tokens = z["tokens"]
            if tokens.ndim == 3:
                N, Hq, Wq = tokens.shape
                C = 1
                tokens = tokens[:, None, :, :]
            elif tokens.ndim == 4:
                N, C, Hq, Wq = tokens.shape
            else:
                raise RuntimeError(f"{token_file}: expected tokens with shape (N,Hq,Wq) or (N,C,Hq,Wq), got {tokens.shape}")

            if self.channel_policy == 'poseidon_matched_4_2':
                if C == 5:
                    tokens = tokens[:, :4]
                    C = 4
                elif C == 3:
                    tokens = tokens[:, :2]
                    C = 2
                else:
                    raise RuntimeError(f'{token_file}: poseidon_matched_4_2 expects 5 or 3 channels, got {C}')

            time_count = int(z["time_count"]) if "time_count" in z else None
            if time_count is None:
                raise RuntimeError(f"{token_file}: missing time_count")
            if N % time_count != 0:
                raise RuntimeError(f"{token_file}: N={N} not divisible by time_count={time_count}")

            n_embed = int(z["n_embed"])
            if n_embed_ref is None:
                n_embed_ref = n_embed
                Hq_ref = int(Hq)
                Wq_ref = int(Wq)
                C_ref = int(C)
            else:
                if n_embed != n_embed_ref or int(Hq) != Hq_ref or int(Wq) != Wq_ref:
                    raise RuntimeError(f"{token_file}: incompatible shape or n_embed")

            if C_max is None:
                C_max = int(C)
            else:
                C_max = max(C_max, int(C))

            file_specs.append((file_i, token_file, tokens, int(C), int(Hq), int(Wq), int(n_embed), int(time_count), int(N)))

        if dataset_slots is not None and len(dataset_slots) != len(file_specs):
            raise RuntimeError(f'dataset_slots has length {len(dataset_slots)}, expected {len(file_specs)}')

        for file_i, token_file, tokens, C, Hq, Wq, n_embed, time_count, N in file_specs:
            arr = tokens.astype(np.int32)
            if C < C_max:
                pad_shape = (N, C_max - C, Hq, Wq)
                pad = np.zeros(pad_shape, dtype=np.int32)
                arr = np.concatenate([arr, pad], axis=1)
            n_traj = N // time_count
            valid_channels = np.zeros((C_max,), dtype=np.int64)
            valid_channels[:C] = 1
            self.arrays.append(arr)
            dataset_slot = int(file_i if dataset_slots is None else dataset_slots[file_i])
            self.metas.append(dict(time_count=time_count, n_traj=n_traj, file=token_file, C=int(C), valid_channels=valid_channels, dataset_slot=dataset_slot))
            for traj_i in range(n_traj):
                for t in range(time_count - 1):
                    for dt in self.delta_t_choices:
                        if (t + dt) < time_count:
                            pairs.append((file_i, traj_i, t, dt))

        self.C = int(C_max)
        self.Hq = int(Hq_ref)
        self.Wq = int(Wq_ref)
        self.n_embed = int(n_embed_ref)
        self.enc_len = self.C * self.Hq * self.Wq

        # Conditioning deck contains [dataset_id, delta_t]. Keep both token
        # families outside the VQ codebook so they can never collide with VQ ids.
        self.cond_offset = self.n_embed
        self.num_dt_tokens = max(self.delta_t_choices) + 1
        self.dataset_offset = self.cond_offset + self.num_dt_tokens
        if num_dataset_tokens is not None and int(num_dataset_tokens) < len(self.token_files):
            raise RuntimeError(f"num_dataset_tokens={num_dataset_tokens} is smaller than token file count={len(self.token_files)}")
        self.num_dataset_tokens = int(num_dataset_tokens) if num_dataset_tokens is not None else len(self.token_files)
        for meta in self.metas:
            if not (0 <= int(meta['dataset_slot']) < self.num_dataset_tokens):
                raise RuntimeError(f'dataset slot {meta["dataset_slot"]} is invalid for num_dataset_tokens={self.num_dataset_tokens}')
        self.bos_id = self.dataset_offset + self.num_dataset_tokens
        self.eos_id = self.bos_id + 1
        self.row_id = self.bos_id + 2
        self.pad_id = self.bos_id + 3
        self.vocab_size = self.bos_id + 4
        self.deck_len = 2
        self.dataset_names = [os.path.basename(os.path.dirname(x)) or os.path.basename(x) for x in self.token_files]
        self.dec_len = self.enc_len

        rows_3d = np.arange(self.Hq, dtype=np.int64).reshape(self.Hq, 1, 1).repeat(self.Wq, axis=1).repeat(self.C, axis=2)
        cols_3d = np.arange(self.Wq, dtype=np.int64).reshape(1, self.Wq, 1).repeat(self.Hq, axis=0).repeat(self.C, axis=2)
        ch_3d = np.arange(self.C, dtype=np.int64).reshape(1, 1, self.C).repeat(self.Hq, axis=0).repeat(self.Wq, axis=1)
        self.rows_flat = rows_3d.reshape(-1)
        self.cols_flat = cols_3d.reshape(-1)
        self.channels_flat = ch_3d.reshape(-1)

        if max_pairs is not None and max_pairs < len(pairs):
            rng = random.Random(self.seed)
            rng.shuffle(pairs)
            pairs = pairs[: int(max_pairs)]
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        file_i, traj_i, t, dt = self.pairs[idx]
        meta = self.metas[file_i]
        tokens = self.arrays[file_i]
        frame0 = traj_i * int(meta['time_count']) + t
        frame1 = frame0 + dt
        tok_t = tokens[frame0]
        tok_tp1 = tokens[frame1]
        valid_channels = meta['valid_channels']

        enc = tok_t.transpose(1, 2, 0).reshape(-1).astype(np.int64)
        tgt = tok_tp1.transpose(1, 2, 0).reshape(-1).astype(np.int64)
        valid_3d = np.broadcast_to(valid_channels.reshape(1, 1, self.C), (self.Hq, self.Wq, self.C))
        valid_flat = valid_3d.reshape(-1).astype(np.int64)
        deck = np.asarray([self.dataset_offset + int(meta['dataset_slot']), self.cond_offset + int(dt)], dtype=np.int64)

        seq = np.empty((self.dec_len + 1,), dtype=np.int64)
        seq[0] = self.bos_id
        seq[1:] = tgt
        dec_in = seq[:-1].copy()
        dec_tgt = seq[1:].copy()

        return (
            torch.from_numpy(deck),
            torch.from_numpy(enc),
            torch.from_numpy(self.rows_flat.copy()),
            torch.from_numpy(self.cols_flat.copy()),
            torch.from_numpy(self.channels_flat.copy()),
            torch.from_numpy(valid_flat),
            torch.from_numpy(dec_in),
            torch.from_numpy(dec_tgt),
            torch.tensor(file_i, dtype=torch.long),
            torch.tensor(dt, dtype=torch.long),
        )


class RMSNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


def _apply_rotary(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Apply RoPE to one even-sized portion of [B, heads, T, head_dim]."""
    rotary_dim = x.shape[-1]
    if rotary_dim % 2:
        raise RuntimeError(f"RoPE dimension must be even, got {rotary_dim}")
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim))
    angles = positions.to(dtype=torch.float32).unsqueeze(-1) * inv_freq
    cos = angles.cos().to(dtype=x.dtype).unsqueeze(1)
    sin = angles.sin().to(dtype=x.dtype).unsqueeze(1)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


def apply_2d_rope(q: torch.Tensor, k: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor):
    """Apply axial RoPE: half a head encodes rows and half encodes columns."""
    if q.shape[-1] % 4:
        raise RuntimeError(f"2D RoPE requires head_dim divisible by 4, got {q.shape[-1]}")
    half = q.shape[-1] // 2
    q = torch.cat((_apply_rotary(q[..., :half], rows), _apply_rotary(q[..., half:], cols)), dim=-1)
    k = torch.cat((_apply_rotary(k[..., :half], rows), _apply_rotary(k[..., half:], cols)), dim=-1)
    return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, use_sdpa: bool, causal: bool):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_sdpa = bool(use_sdpa)
        self.causal = bool(causal)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, rows: torch.Tensor, cols: torch.Tensor, key_padding_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_2d_rope(q, k, rows, cols)
        attention_mask = None
        if key_padding_mask is not None:
            attention_mask = ~key_padding_mask[:, None, None, :]
            if self.causal:
                attention_mask = attention_mask & torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=self.causal and attention_mask is None)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                att = att.masked_fill(~attention_mask, float('-inf'))
            if self.causal:
                mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
                att = att.masked_fill(mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class CrossAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, use_sdpa: bool):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_sdpa = bool(use_sdpa)
        self.q = nn.Linear(n_embd, n_embd)
        self.kv = nn.Linear(n_embd, 2 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, mem, query_rows: torch.Tensor, query_cols: torch.Tensor, memory_rows: torch.Tensor, memory_cols: torch.Tensor, memory_padding_mask=None):
        B, T, C = x.shape
        _, S, _ = mem.shape
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.kv(mem)
        k, v = kv.split(C, dim=2)
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        q_row, _ = apply_2d_rope(q, q, query_rows, query_cols)
        k_row, _ = apply_2d_rope(k, k, memory_rows, memory_cols)
        q, k = q_row, k_row
        attention_mask = None if memory_padding_mask is None else ~memory_padding_mask[:, None, None, :]
        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                att = att.masked_fill(~attention_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, mlp_hidden_dim, dropout, use_sdpa):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, use_sdpa=use_sdpa, causal=False)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = SwiGLU(n_embd, mlp_hidden_dim, dropout)

    def forward(self, x, rows: torch.Tensor, cols: torch.Tensor, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), rows, cols, key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, mlp_hidden_dim, dropout, use_sdpa: bool):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.self_attn = CausalSelfAttention(n_embd, n_head, dropout, use_sdpa=use_sdpa, causal=True)
        self.ln2 = RMSNorm(n_embd)
        self.cross_attn = CrossAttention(n_embd, n_head, dropout, use_sdpa=use_sdpa)
        self.ln3 = RMSNorm(n_embd)
        self.mlp = SwiGLU(n_embd, mlp_hidden_dim, dropout)

    def forward(self, x, rows: torch.Tensor, cols: torch.Tensor, mem, memory_rows: torch.Tensor, memory_cols: torch.Tensor,
                key_padding_mask=None, memory_padding_mask=None):
        x = x + self.self_attn(self.ln1(x), rows, cols, key_padding_mask)
        x = x + self.cross_attn(self.ln2(x), mem, rows, cols, memory_rows, memory_cols, memory_padding_mask)
        x = x + self.mlp(self.ln3(x))
        return x


class EncDecSharedGPT(nn.Module):
    def __init__(self, vocab_size: int, n_embed: int, Hq: int, Wq: int, C: int, deck_len: int, dec_block_size: int, n_layer_enc: int, n_layer_dec: int, n_head: int, n_embd: int, mlp_hidden_dim: int, dropout: float, pad_id: int, use_sdpa: bool):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_embed = int(n_embed)
        self.Hq, self.Wq, self.C = int(Hq), int(Wq), int(C)
        self.enc_len_frame = self.C * self.Hq * self.Wq
        self.deck_len = int(deck_len)
        self.dec_block_size = int(dec_block_size)
        self.pad_id = int(pad_id)
        self.use_sdpa = bool(use_sdpa)

        self.tok_emb = nn.Embedding(self.vocab_size, n_embd)
        self.row_emb = nn.Embedding(self.Hq, n_embd)
        self.col_emb = nn.Embedding(self.Wq, n_embd)
        self.channel_emb = nn.Embedding(self.C, n_embd)
        self.deck_type = nn.Parameter(torch.zeros(1, 1, n_embd)) if self.deck_len > 0 else None
        self.frame_type = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.drop = nn.Dropout(dropout)

        self.enc_blocks = nn.ModuleList([EncoderBlock(n_embd, n_head, mlp_hidden_dim, dropout, use_sdpa=use_sdpa) for _ in range(n_layer_enc)])
        self.dec_blocks = nn.ModuleList([DecoderBlock(n_embd, n_head, mlp_hidden_dim, dropout, use_sdpa=use_sdpa) for _ in range(n_layer_dec)])
        self.enc_ln = RMSNorm(n_embd)
        self.dec_ln = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, self.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, enc_tokens: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, channels: torch.Tensor, deck_tokens: torch.Tensor = None) -> torch.Tensor:
        B, L = enc_tokens.shape
        assert L == self.enc_len_frame
        xf = self.tok_emb(enc_tokens) + self.row_emb(rows) + self.col_emb(cols) + self.channel_emb(channels) + self.frame_type
        memory_rows, memory_cols = rows, cols
        memory_padding_mask = enc_tokens.eq(self.pad_id)
        parts = []
        if deck_tokens is not None and self.deck_len > 0:
            xd = self.tok_emb(deck_tokens)
            xd = xd + self.deck_type
            parts.append(xd)
            zeros = torch.zeros((B, self.deck_len), dtype=rows.dtype, device=rows.device)
            memory_rows = torch.cat((zeros, rows), dim=1)
            memory_cols = torch.cat((zeros, cols), dim=1)
            memory_padding_mask = torch.cat((torch.zeros_like(deck_tokens, dtype=torch.bool), memory_padding_mask), dim=1)
        parts.append(xf)
        x = self.drop(torch.cat(parts, dim=1))
        for block in self.enc_blocks:
            x = block(x, memory_rows, memory_cols, memory_padding_mask)
        self.memory_rows = memory_rows
        self.memory_cols = memory_cols
        self.memory_padding_mask = memory_padding_mask
        return self.enc_ln(x)

    def decode(self, dec_in: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, channels: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        B, T = dec_in.shape
        rows, cols, channels = rows[:, :T], cols[:, :T], channels[:, :T]
        x = self.tok_emb(dec_in) + self.row_emb(rows) + self.col_emb(cols) + self.channel_emb(channels)
        x = self.drop(x)
        for block in self.dec_blocks:
            x = block(x, rows, cols, mem, self.memory_rows, self.memory_cols, dec_in.eq(self.pad_id), self.memory_padding_mask)
        x = self.dec_ln(x)
        return self.lm_head(x)

    def forward(self, enc_tokens: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, channels: torch.Tensor, dec_in: torch.Tensor, deck_tokens: torch.Tensor = None):
        mem = self.encode(enc_tokens, rows, cols, channels, deck_tokens=deck_tokens)
        return self.decode(dec_in, rows, cols, channels, mem)

    def loss_masked(self, logits: torch.Tensor, dec_tgt: torch.Tensor, valid_flat: torch.Tensor):
        per_tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            dec_tgt.reshape(-1),
            ignore_index=self.pad_id,
            reduction='none',
        ).view_as(dec_tgt)
        mask = valid_flat.to(per_tok.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (per_tok * mask).sum() / denom


def save_ckpt(path, model, opt, scaler, step, best_val, args, extra: dict, training_state=None):
    raw = model.module if hasattr(model, 'module') else model
    ck = {
        'model': raw.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'step': int(step),
        'best_val': float(best_val),
        'cfg': vars(args),
        'training_state': training_state,
    }
    ck.update(extra)
    if getattr(args, 'phaedra_ar_df4', False):
        ck.update(model_type='phaedra_ar_df4', autoregressive=True)
    if getattr(args, 'ar_df4_transfer', False):
        ck.update(model_type='phaedra_ar_df4_transfer', autoregressive=True)
    temporary_path = f"{path}.tmp"
    torch.save(ck, temporary_path)
    os.replace(temporary_path, path)


def load_ckpt(model, path, opt=None, scaler=None):
    ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    raw = model.module if hasattr(model, 'module') else model
    missing, unexpected = raw.load_state_dict(ck['model'], strict=False)
    if opt is not None and 'opt' in ck and ck['opt'] is not None:
        opt.load_state_dict(ck['opt'])
    if scaler is not None and 'scaler' in ck and ck['scaler'] is not None:
        scaler.load_state_dict(ck['scaler'])
    return ck, missing, unexpected


def lr_at_step(base_lr, step, warmup_steps, total_steps, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)


def eval_loss(model, dl, device, dataset_names, delta_t_choices, amp=False):
    """Return globally reduced token-weighted CE plus per-dataset and per-dt CE."""
    model.eval()
    n_ds = len(dataset_names)
    max_dt = max(delta_t_choices)
    # [loss_sum, token_count] for global, dataset, and dt buckets.
    global_stats = torch.zeros(2, dtype=torch.float64, device=device)
    ds_stats = torch.zeros((n_ds, 2), dtype=torch.float64, device=device)
    dt_stats = torch.zeros((max_dt + 1, 2), dtype=torch.float64, device=device)

    with torch.no_grad():
        for deck, enc, rows, cols, channels, valid_flat, dec_in, dec_tgt, file_i, dt_val in dl:
            deck = deck.to(device, non_blocking=True)
            enc = enc.to(device, non_blocking=True)
            rows = rows.to(device, non_blocking=True)
            cols = cols.to(device, non_blocking=True)
            channels = channels.to(device, non_blocking=True)
            valid_flat = valid_flat.to(device, non_blocking=True)
            dec_in = dec_in.to(device, non_blocking=True)
            dec_tgt = dec_tgt.to(device, non_blocking=True)
            file_i = file_i.to(device, non_blocking=True)
            dt_val = dt_val.to(device, non_blocking=True)

            evaluation_model = model.module if amp and hasattr(model, 'module') else model
            with autocast_ctx(amp and device.type == 'cuda'):
                logits = evaluation_model(enc, rows, cols, channels, dec_in, deck_tokens=deck)
            if amp:
                logits = logits.float()
            per_tok = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                dec_tgt.reshape(-1),
                reduction='none',
            ).view_as(dec_tgt)
            mask = valid_flat.to(per_tok.dtype)

            # Per-example sums/counts make the grouped reductions exact even when
            # datasets have different numbers of valid channels.
            ex_loss = (per_tok * mask).sum(dim=1).double()
            ex_count = mask.sum(dim=1).double()
            global_stats[0] += ex_loss.sum()
            global_stats[1] += ex_count.sum()
            for b in range(enc.size(0)):
                dsi = int(file_i[b].item())
                dti = int(dt_val[b].item())
                ds_stats[dsi, 0] += ex_loss[b]
                ds_stats[dsi, 1] += ex_count[b]
                dt_stats[dti, 0] += ex_loss[b]
                dt_stats[dti, 1] += ex_count[b]

    if ddp_is_enabled():
        dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(ds_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(dt_stats, op=dist.ReduceOp.SUM)

    global_ce = (global_stats[0] / global_stats[1].clamp_min(1.0)).item()
    per_dataset = {
        dataset_names[i]: (ds_stats[i, 0] / ds_stats[i, 1].clamp_min(1.0)).item()
        for i in range(n_ds) if ds_stats[i, 1].item() > 0
    }
    per_dt = {
        dt: (dt_stats[dt, 0] / dt_stats[dt, 1].clamp_min(1.0)).item()
        for dt in delta_t_choices if dt_stats[dt, 1].item() > 0
    }
    model.train()
    return float(global_ce), per_dataset, per_dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tokens", nargs='+', required=True)
    ap.add_argument("--val_tokens", nargs='+', required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--continuation_parent", type=str, default=None)
    ap.add_argument("--ce_continuation_parent", type=str, default=None)
    ap.add_argument("--phaedra_monitor_config", type=str, default=None)
    ap.add_argument("--phaedra_forecast_plateau", action="store_true")
    ap.add_argument("--transfer_manifest", type=str, default=None)
    ap.add_argument("--ar583_transfer", action="store_true", default=None)
    ap.add_argument("--ar_df4_transfer", action="store_true", default=None)
    ap.add_argument("--ar_df4_transfer_16gpu", action="store_true", default=None)
    ap.add_argument("--cerm_transfer", action="store_true", default=None)
    ap.add_argument("--phaedra_ar_df4", action="store_true", default=None)
    ap.add_argument("--init_from", type=str, default=None)
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--num_dataset_tokens", type=int, default=None)
    ap.add_argument("--dataset_slots", type=int, nargs='+', default=None)
    ap.add_argument("--channel_policy", choices=['all', 'poseidon_matched_4_2'], default='all')
    ap.add_argument("--delta_t_choices", type=int, nargs='+', default=[1])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=100000, help="-1 enables guarded plateau-only Phaedra training")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_at_start", action="store_true")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--early_stop_patience_evals", type=int, default=0)
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=3000)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--n_layer_enc", type=int, default=6)
    ap.add_argument("--n_layer_dec", type=int, default=12)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=768)
    ap.add_argument("--mlp_hidden_dim", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--use_sdpa", action="store_true")
    args = ap.parse_args()
    if args.cerm_transfer and not args.ar583_transfer:
        ap.error("CE-RM transfer requires the audited AR583 transfer mode")
    if args.grad_accum < 1: raise ValueError("grad_accum must be positive")
    if args.ar583_transfer and (not args.transfer_manifest or args.ce_continuation_parent or args.phaedra_forecast_plateau):
        ap.error("AR583 fine-tuning requires a transfer manifest and CE-only downstream stopping")
    if args.ar_df4_transfer and (not args.transfer_manifest or args.phaedra_ar_df4 or args.ar583_transfer
                                or args.ce_continuation_parent or args.phaedra_forecast_plateau):
        ap.error('AR df4 transfer requires its own manifest and downstream stopping')
    if args.ar_df4_transfer_16gpu and not args.ar_df4_transfer:
        raise ValueError('Sixteen-GPU transfer requires AR df4 transfer mode')
    track_rng = bool(args.ce_continuation_parent or args.ar583_transfer or args.phaedra_ar_df4 or args.ar_df4_transfer)
    if args.phaedra_ar_df4:
        from scripts.phaedra_ar_df4_pretraining import validate_pretraining as validate_ar_df4
        validate_ar_df4(args)
    if args.continuation_parent and not args.resume:
        raise ValueError("Explicit continuation requires --resume")
    if args.transfer_manifest and (not (args.init_from or args.resume) or args.continuation_parent or args.phaedra_monitor_config):
        raise ValueError("Transfer requires initialization/resume without source-continuation monitoring")
    if args.phaedra_forecast_plateau:
        from scripts.phaedra_pretraining import validate_pretraining
        validate_pretraining(args)
    if args.ce_continuation_parent:
        from scripts.phaedra_ar_ce_continuation import validate_continuation as validate_ar_ce, record_continuation as record_ar_ce
        validate_ar_ce(args)
    if not (args.phaedra_forecast_plateau or args.phaedra_ar_df4) and (args.max_steps < -1 or (args.max_steps == -1 and (
            not (args.continuation_parent or args.transfer_manifest or args.ce_continuation_parent) or args.early_stop_patience_evals <= 0
            or args.warmup_steps != 0 or args.min_lr_ratio != 1.0))):
        raise ValueError("Uncapped training requires constant-LR continuation with early stopping")
    if args.phaedra_monitor_config and ((not args.resume and not args.phaedra_forecast_plateau) or args.eval_every % args.save_every):
        raise ValueError("Generated monitoring requires resume and a checkpoint at every evaluation")

    os.makedirs(args.out_dir, exist_ok=True)
    rank, world_size, local_rank, cuda_device = ddp_setup()
    if args.phaedra_ar_df4:
        validate_ar_df4(args, world_size)
    if args.phaedra_forecast_plateau:
        validate_pretraining(args, world_size)
    if args.ce_continuation_parent:
        validate_ar_ce(args, world_size)
    transfer_ranks = 16 if args.ar_df4_transfer_16gpu else (4 if args.cerm_transfer else 8)
    if args.transfer_manifest and world_size != transfer_ranks:
        raise ValueError("Audited Phaedra transfer requires its configured rank count")
    if args.ar_df4_transfer_16gpu:
        from scripts.phaedra_ar_df4_transfer import audit_sixteen_gpus
        audit_sixteen_gpus()
    seed_all(args.seed + rank)
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")

    try:
        if args.resume and args.init_from:
            raise RuntimeError("--resume and --init_from are mutually exclusive")

        ds_tr = SharedFixedTokenDataset(args.train_tokens, max_pairs=args.max_pairs, seed=args.seed, delta_t_choices=args.delta_t_choices, num_dataset_tokens=args.num_dataset_tokens, dataset_slots=args.dataset_slots, channel_policy=args.channel_policy)
        ds_va = SharedFixedTokenDataset(args.val_tokens, max_pairs=None, seed=args.seed, delta_t_choices=args.delta_t_choices, num_dataset_tokens=args.num_dataset_tokens, dataset_slots=args.dataset_slots, channel_policy=args.channel_policy)

        assert ds_tr.Hq == ds_va.Hq and ds_tr.Wq == ds_va.Wq and ds_tr.C == ds_va.C
        assert ds_tr.n_embed == ds_va.n_embed

        if args.phaedra_ar_df4:
            from scripts.phaedra_ar_df4_pretraining import audit_datasets
            audit_datasets(args, ds_tr, ds_va)

        if args.ar583_transfer:
            from scripts.phaedra_transfer import audit_ar583_datasets
            audit_ar583_datasets(args, ds_tr, ds_va)
        if args.ar_df4_transfer:
            from scripts.phaedra_ar_df4_transfer import audit_datasets
            audit_datasets(args, ds_tr, ds_va)

        batch_sampler = ResumableDistributedBatchSampler(ds_tr, args.batch_size, world_size, rank, args.seed)
        va_sampler = DistributedEvalSampler(ds_va, rank, world_size) if ddp_is_enabled() else None

        dl_tr = DataLoader(ds_tr, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True,
                           generator=torch.Generator().manual_seed(args.seed))
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, sampler=va_sampler, num_workers=args.num_workers, pin_memory=True,
                           generator=torch.Generator().manual_seed(args.seed + 1) if args.ar583_transfer or args.ar_df4_transfer else None)

        mlp_hidden_dim = int(args.mlp_hidden_dim) if args.mlp_hidden_dim else int(round((8 * args.n_embd / 3) / 256) * 256)
        model = EncDecSharedGPT(
            vocab_size=ds_tr.vocab_size,
            n_embed=ds_tr.n_embed,
            Hq=ds_tr.Hq,
            Wq=ds_tr.Wq,
            C=ds_tr.C,
            deck_len=ds_tr.deck_len,
            dec_block_size=ds_tr.dec_len,
            n_layer_enc=args.n_layer_enc,
            n_layer_dec=args.n_layer_dec,
            n_head=args.n_head,
            n_embd=args.n_embd,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=args.dropout,
            pad_id=ds_tr.pad_id,
            use_sdpa=args.use_sdpa,
        ).to(device)

        if ddp_is_enabled():
            model = nn.parallel.DistributedDataParallel(model, device_ids=[cuda_device], output_device=cuda_device, find_unused_parameters=False)

        opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        use_amp = torch.cuda.is_available()
        scaler = make_scaler(use_amp)

        best_val = float('inf')
        step = 0
        micro_step = 0
        evals_since_best = 0
        from scripts.phaedra_pretraining import initial_forecast_state, update_forecast_state, link_checkpoint
        forecast_state = initial_forecast_state()
        first_ce_continuation = False
        checkpoint_rng = None
        monitor_checkpoint = args.resume

        if args.resume:
            ck, missing, unexpected = load_ckpt(model, args.resume, opt=opt, scaler=scaler)
            if args.phaedra_ar_df4:
                validate_ar_df4(args, world_size, ck, missing, unexpected)
            if args.phaedra_forecast_plateau:
                validate_pretraining(args, world_size, ck, missing, unexpected)
            if args.ce_continuation_parent:
                first_ce_continuation = validate_ar_ce(args, world_size, ck, missing, unexpected)
                if is_rank0(rank):
                    record_ar_ce(args, ck, first_ce_continuation)
            if args.transfer_manifest:
                from scripts.phaedra_transfer import validate_transfer
                validate_transfer(args, ck, missing, unexpected)
            if args.continuation_parent:
                from scripts.phaedra_continuation import record_continuation, validate_continuation
                validate_continuation(args, ck, missing, unexpected)
                if is_rank0(rank):
                    record_continuation(args)
            step = int(ck.get("step", 0))
            micro_step = step * args.grad_accum
            best_val = float(ck.get("best_val", best_val))
            training_state = ck.get("training_state")
            if training_state is not None:
                if training_state["grad_accum"] != args.grad_accum:
                    raise ValueError("Cannot change grad_accum during data-stream resume")
                batch_sampler.load_state_dict(training_state["data_stream"])
                evals_since_best = int(training_state["evals_since_best"])
                forecast_state = training_state.get("forecast_plateau", forecast_state)
                if track_rng:
                    checkpoint_rng = training_state.get("rng_states")
                    if first_ce_continuation:
                        evals_since_best = 0
            else:
                batch_sampler.seek(micro_step)
                if is_rank0(rank): print(">> legacy checkpoint: inferring data cursor from optimizer step; prior replay history cannot be recovered", flush=True)
            if is_rank0(rank):
                print(f">> RESUME {args.resume} missing={len(missing)} unexpected={len(unexpected)} start_step={step} best_val={best_val:.6f}", flush=True)
                print(f">> data_resume epoch={batch_sampler.epoch} batch_offset={batch_sampler.batch_offset}", flush=True)
            del ck
        elif args.init_from:
            initialized, missing, unexpected = load_ckpt(model, args.init_from)
            if args.transfer_manifest:
                from scripts.phaedra_transfer import validate_transfer
                validate_transfer(args, initialized, missing, unexpected)
            del initialized
            if is_rank0(rank):
                print(f">> INIT_FROM {args.init_from} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

        if is_rank0(rank):
            print(f">> train_pairs={len(ds_tr)} val_pairs={len(ds_va)} channel_policy={args.channel_policy} C={ds_tr.C} Hq={ds_tr.Hq} Wq={ds_tr.Wq} dec_len={ds_tr.dec_len} deck_len={ds_tr.deck_len} n_embed={ds_tr.n_embed} vocab={ds_tr.vocab_size} datasets={ds_tr.dataset_names} delta_t_choices={ds_tr.delta_t_choices}", flush=True)
            print(f">> architecture=llama2drope rmsnorm=1 swiglu=1 mlp_hidden_dim={mlp_hidden_dim}", flush=True)
            print(f">> params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
            if args.phaedra_ar_df4:
                print(f'>> AR df4 causal_decoder=1 world_size={world_size} batch_per_rank={args.batch_size} '
                      f'grad_accum={args.grad_accum} global_batch={world_size * args.batch_size * args.grad_accum} '
                      'stopping_metric=full_validation_token_ce patience=10 eval_every=1000 recovery_every=100', flush=True)

        if track_rng:
            from scripts.parallel_phaedra_support import collect_rng_states, restore_rng_state
            from scripts.parallel_ce_continuation import checkpoint_metadata
            if checkpoint_rng is None:
                checkpoint_rng = collect_rng_states()
            if first_ce_continuation:
                monitor_checkpoint = os.path.join(args.out_dir, f"gpt_step{step}.pt")
                if is_rank0(rank):
                    save_ckpt(monitor_checkpoint, model, opt, scaler, step, best_val, args, checkpoint_metadata(ds_tr),
                              training_state=dict(data_stream=batch_sampler.state_dict(), grad_accum=args.grad_accum,
                                  evals_since_best=evals_since_best, forecast_plateau=forecast_state, rng_states=checkpoint_rng))
                    link_checkpoint(monitor_checkpoint, os.path.join(args.out_dir, "gpt_best.pt"))
                    print(">> inherited CE-best baseline; optimizer/data cursor retained; parent RNG unavailable, reseeded", flush=True)
                if ddp_is_enabled():
                    dist.barrier()
        t0 = time.time()
        if args.eval_at_start and not args.resume:
            baseline, val_by_dataset, val_by_dt = eval_loss(model, dl_va, device, ds_va.dataset_names, ds_va.delta_t_choices,
                                                          **(dict(amp=True) if args.ar_df4_transfer else {}))
            if not math.isfinite(baseline):
                raise RuntimeError("Initial validation loss is not finite")
            best_val = baseline
            if is_rank0(rank):
                print(f"[{step:>7}] baseline=True val_loss={baseline:.6f} ({(time.time() - t0) / 60.0:.1f} min)", flush=True)
                print(">> val_by_dataset " + " ".join(f"{key}={value:.4f}" for key, value in val_by_dataset.items()), flush=True)
                print(">> val_by_dt " + " ".join(f"dt{key}={value:.4f}" for key, value in val_by_dt.items()), flush=True)
                extra = dict(vocab_size=int(ds_tr.vocab_size), n_embed=int(ds_tr.n_embed), C=int(ds_tr.C), Hq=int(ds_tr.Hq), Wq=int(ds_tr.Wq), dec_block=int(ds_tr.dec_len), bos_id=int(ds_tr.bos_id), eos_id=int(ds_tr.eos_id), row_id=int(ds_tr.row_id), pad_id=int(ds_tr.pad_id), deck_len=int(ds_tr.deck_len), dataset_offset=int(ds_tr.dataset_offset), num_dataset_tokens=int(ds_tr.num_dataset_tokens), dataset_names=list(ds_tr.dataset_names), delta_t_choices=list(ds_tr.delta_t_choices), token_order='spatial_major', max_channels=int(ds_tr.C), max_Hq=int(ds_tr.Hq), max_Wq=int(ds_tr.Wq), max_dec_len=int(ds_tr.dec_len))
                save_ckpt(os.path.join(args.out_dir, "gpt_best.pt"), model, opt, scaler, step, best_val, args, extra,
                          training_state=dict(data_stream=batch_sampler.state_dict(), grad_accum=args.grad_accum, evals_since_best=evals_since_best,
                                              **(dict(rng_states=checkpoint_rng) if track_rng else {})))
                print(">> saved gpt_best.pt (initial validation baseline)", flush=True)
            if ddp_is_enabled():
                dist.barrier()
        start_step = step
        from scripts.phaedra_continuation import plateau_reached, training_finished
        should_stop = plateau_reached(args.early_stop_patience_evals,
                                      forecast_state["stale"] if args.phaedra_forecast_plateau else evals_since_best)
        if is_rank0(rank):
            print(f">> stopping max_steps={args.max_steps} patience={args.early_stop_patience_evals} stale={evals_since_best} min_delta={args.early_stop_min_delta}", flush=True)
            if args.ce_continuation_parent:
                print(">> stopping_metric=full_validation_token_ce patience_evals=10 patience_steps=10000 forecast_metrics=diagnostic_only", flush=True)
            elif args.ar583_transfer or args.ar_df4_transfer:
                print(">> stopping_metric=full_validation_token_ce patience_evals=10 patience_steps=1000 forecast_metrics=disabled", flush=True)
        monitor = None
        def run_forecast_validation(checkpoint_path):
            nonlocal forecast_state, should_stop
            metrics = monitor.run(model, step, checkpoint_path)
            if args.ce_continuation_parent and is_rank0(rank):
                from scripts.phaedra_ar_ce_continuation import inherit_forecast_selection
                inherit_forecast_selection(args, monitor)
            if args.phaedra_forecast_plateau:
                forecast_state, improved = update_forecast_state(
                    forecast_state, metrics["mean_ratio_dt20"], step,
                    args.warmup_steps, args.early_stop_min_delta)
                should_stop = plateau_reached(args.early_stop_patience_evals, forecast_state["stale"])
                if is_rank0(rank):
                    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
                    saved["training_state"]["forecast_plateau"] = forecast_state
                    temporary = f"{checkpoint_path}.tmp"
                    torch.save(saved, temporary)
                    os.replace(temporary, checkpoint_path)
                    del saved
                    if improved:
                        link_checkpoint(checkpoint_path, os.path.join(args.out_dir, "gpt_generated_best.pt"))
                    print(f">> forecast_plateau metric=mean_ratio_dt20 best={forecast_state['best']:.7g} "
                          f"stale={forecast_state['stale']}/{args.early_stop_patience_evals} stop={should_stop}", flush=True)
            if ddp_is_enabled():
                dist.barrier()

        if args.phaedra_monitor_config:
            from scripts.phaedra_forecast_monitor import PhaedraForecastMonitor
            monitor = PhaedraForecastMonitor(args, ds_va, device, rank, world_size)
            if args.ce_continuation_parent and is_rank0(rank):
                from scripts.phaedra_ar_ce_continuation import inherit_forecast_selection
                inherit_forecast_selection(args, monitor)
            if args.ce_continuation_parent and ddp_is_enabled():
                dist.barrier()
            if not args.phaedra_forecast_plateau or (step > forecast_state["last_step"] and step % args.eval_every == 0):
                run_forecast_validation(monitor_checkpoint)
            if ddp_is_enabled():
                dist.barrier()
        if track_rng:
            if args.ar_df4_transfer_16gpu and checkpoint_rng[rank] is None:
                seed_all(args.seed + 1000003 + step * world_size + rank)
            else:
                restore_rng_state(checkpoint_rng[rank])
            if args.ar_df4_transfer_16gpu:
                checkpoint_rng = collect_rng_states()
                if is_rank0(rank):
                    print(f'>> AR df4 transfer world_size={world_size} batch_per_rank={args.batch_size} '
                          f'grad_accum={args.grad_accum} global_batch={world_size * args.batch_size * args.grad_accum} '
                          'optimizer/scaler/data/patience restored; all-rank RNG ready', flush=True)
        while not training_finished(step, args.max_steps, should_stop):
            for deck, enc, rows, cols, channels, valid_flat, dec_in, dec_tgt, file_i, dt_val in dl_tr:
                deck = deck.to(device, non_blocking=True)
                enc = enc.to(device, non_blocking=True)
                rows = rows.to(device, non_blocking=True)
                cols = cols.to(device, non_blocking=True)
                channels = channels.to(device, non_blocking=True)
                valid_flat = valid_flat.to(device, non_blocking=True)
                dec_in = dec_in.to(device, non_blocking=True)
                dec_tgt = dec_tgt.to(device, non_blocking=True)

                lr = lr_at_step(args.lr, step, args.warmup_steps, args.max_steps, min_lr_ratio=args.min_lr_ratio)
                for pg in opt.param_groups:
                    pg["lr"] = lr

                if micro_step % args.grad_accum == 0:
                    opt.zero_grad(set_to_none=True)

                # Avoid an all-reduce on the first grad_accum-1 microsteps.
                will_step = ((micro_step + 1) % args.grad_accum == 0)
                sync_ctx = nullcontext() if (not ddp_is_enabled() or will_step) else model.no_sync()
                with sync_ctx:
                    with autocast_ctx(use_amp):
                        logits = model(enc, rows, cols, channels, dec_in, deck_tokens=deck)
                        base = model.module if hasattr(model, 'module') else model
                        loss = base.loss_masked(logits, dec_tgt, valid_flat)
                        loss = loss / args.grad_accum
                    scaler.scale(loss).backward()
                micro_step += 1
                batch_sampler.advance()

                if micro_step % args.grad_accum == 0:
                    scaler.unscale_(opt)
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    step += 1
                    if is_rank0(rank) and (step == start_step + 1 or step % 100 == 0):
                        print(f">> step={step} lr={lr:.3e} train_loss={float(loss.item()) * args.grad_accum:.6f} data_epoch={batch_sampler.epoch} data_batch={batch_sampler.batch_offset}", flush=True)

                    if step % args.eval_every == 0:
                        if ddp_is_enabled():
                            dist.barrier()
                        val, val_by_dataset, val_by_dt = eval_loss(model, dl_va, device, ds_va.dataset_names, ds_va.delta_t_choices,
                                                                  **(dict(amp=True) if args.phaedra_ar_df4 or args.ar_df4_transfer else {}))
                        if track_rng:
                            if not math.isfinite(val):
                                raise RuntimeError("Nonfinite continuation validation CE")
                            checkpoint_rng = collect_rng_states()
                        if is_rank0(rank):
                            elapsed_min = (time.time() - t0) / 60.0
                            print(f"[{step:>7}] lr={lr:.3e} train_loss={float(loss.item())*args.grad_accum:.4f} val_loss={val:.4f} ({elapsed_min:.1f} min)", flush=True)
                            print(">> val_by_dataset " + " ".join(f"{k}={v:.4f}" for k, v in val_by_dataset.items()), flush=True)
                            print(">> val_by_dt " + " ".join(f"dt{k}={v:.4f}" for k, v in val_by_dt.items()), flush=True)
                            if val < (best_val - float(args.early_stop_min_delta)):
                                best_val = val
                                if not args.phaedra_forecast_plateau:
                                    evals_since_best = 0
                                extra = dict(vocab_size=int(ds_tr.vocab_size), n_embed=int(ds_tr.n_embed), C=int(ds_tr.C), Hq=int(ds_tr.Hq), Wq=int(ds_tr.Wq), dec_block=int(ds_tr.dec_len), bos_id=int(ds_tr.bos_id), eos_id=int(ds_tr.eos_id), row_id=int(ds_tr.row_id), pad_id=int(ds_tr.pad_id), deck_len=int(ds_tr.deck_len), dataset_offset=int(ds_tr.dataset_offset), num_dataset_tokens=int(ds_tr.num_dataset_tokens), dataset_names=list(ds_tr.dataset_names), delta_t_choices=list(ds_tr.delta_t_choices), token_order='spatial_major', max_channels=int(ds_tr.C), max_Hq=int(ds_tr.Hq), max_Wq=int(ds_tr.Wq), max_dec_len=int(ds_tr.dec_len))
                                save_ckpt(os.path.join(args.out_dir, "gpt_best.pt"), model, opt, scaler, step, best_val, args, extra,
                                          training_state=dict(data_stream=batch_sampler.state_dict(), grad_accum=args.grad_accum, evals_since_best=evals_since_best, forecast_plateau=forecast_state,
                                              **(dict(rng_states=checkpoint_rng) if track_rng else {})))
                                print(">> saved gpt_best.pt", flush=True)
                            elif not args.phaedra_forecast_plateau:
                                evals_since_best += 1
                                if args.early_stop_patience_evals > 0 and evals_since_best >= int(args.early_stop_patience_evals):
                                    print(f">> early stopping triggered: evals_since_best={evals_since_best} best_val={best_val:.6f}", flush=True)
                                    should_stop = True
                            if track_rng:
                                print(f">> ce_plateau best_ce={best_val:.9f} stale_evals={evals_since_best}/10 stop={should_stop}", flush=True)
                        if ddp_is_enabled():
                            stop_flag = torch.tensor(int(should_stop), device=device)
                            dist.broadcast(stop_flag, src=0)
                            should_stop = bool(stop_flag.item())
                            dist.barrier()

                    if step % args.save_every == 0:
                        if track_rng:
                            checkpoint_rng = collect_rng_states()
                        if is_rank0(rank):
                            extra = dict(vocab_size=int(ds_tr.vocab_size), n_embed=int(ds_tr.n_embed), C=int(ds_tr.C), Hq=int(ds_tr.Hq), Wq=int(ds_tr.Wq), dec_block=int(ds_tr.dec_len), bos_id=int(ds_tr.bos_id), eos_id=int(ds_tr.eos_id), row_id=int(ds_tr.row_id), pad_id=int(ds_tr.pad_id), deck_len=int(ds_tr.deck_len), dataset_offset=int(ds_tr.dataset_offset), num_dataset_tokens=int(ds_tr.num_dataset_tokens), dataset_names=list(ds_tr.dataset_names), delta_t_choices=list(ds_tr.delta_t_choices), token_order='spatial_major', max_channels=int(ds_tr.C), max_Hq=int(ds_tr.Hq), max_Wq=int(ds_tr.Wq), max_dec_len=int(ds_tr.dec_len))
                            checkpoint_name = 'gpt_resume.pt' if args.phaedra_ar_df4 else f'gpt_step{step}.pt'
                            save_ckpt(os.path.join(args.out_dir, checkpoint_name), model, opt, scaler, step, best_val, args, extra,
                                      training_state=dict(data_stream=batch_sampler.state_dict(), grad_accum=args.grad_accum, evals_since_best=evals_since_best, forecast_plateau=forecast_state,
                                          **(dict(rng_states=checkpoint_rng) if track_rng else {})))
                            print(f">> saved {checkpoint_name} step={step}", flush=True)

                    if monitor is not None and step % args.eval_every == 0:
                        if ddp_is_enabled():
                            dist.barrier()
                        run_forecast_validation(os.path.join(args.out_dir, f"gpt_step{step}.pt"))
                        if ddp_is_enabled():
                            dist.barrier()

                    if training_finished(step, args.max_steps, should_stop):
                        break

        if track_rng:
            checkpoint_rng = collect_rng_states()
        if is_rank0(rank):
            extra = dict(vocab_size=int(ds_tr.vocab_size), n_embed=int(ds_tr.n_embed), C=int(ds_tr.C), Hq=int(ds_tr.Hq), Wq=int(ds_tr.Wq), dec_block=int(ds_tr.dec_len), bos_id=int(ds_tr.bos_id), eos_id=int(ds_tr.eos_id), row_id=int(ds_tr.row_id), pad_id=int(ds_tr.pad_id), deck_len=int(ds_tr.deck_len), dataset_offset=int(ds_tr.dataset_offset), num_dataset_tokens=int(ds_tr.num_dataset_tokens), dataset_names=list(ds_tr.dataset_names), delta_t_choices=list(ds_tr.delta_t_choices), token_order='spatial_major', max_channels=int(ds_tr.C), max_Hq=int(ds_tr.Hq), max_Wq=int(ds_tr.Wq), max_dec_len=int(ds_tr.dec_len))
            save_ckpt(os.path.join(args.out_dir, "gpt_final.pt"), model, opt, scaler, step, best_val, args, extra,
                      training_state=dict(data_stream=batch_sampler.state_dict(), grad_accum=args.grad_accum, evals_since_best=evals_since_best, forecast_plateau=forecast_state,
                          **(dict(rng_states=checkpoint_rng) if track_rng else {})))
            print(">> training complete", flush=True)
    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()

"""Named-field adaptation of the existing AR583 decoder without token renumbering."""
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.checkpoint import checkpoint

from scripts.train_encdec_vq_transformer_pdegym6_llama import EncDecSharedGPT
from scripts.llama_cached_inference import CachedDecoder
from .conditioning import ConditioningPrefix

ARCHITECTURE = dict(n_layer_enc=12, n_layer_dec=24, n_head=16, n_embd=1024, mlp_hidden_dim=2816)
MORPHOLOGY, AMPLITUDE, VOCABULARY = 8640, 1024, 9695
BOS, PAD = 9691, 9694


def field_slots(fields):
    known = dict(density=0, velocity_x=1, velocity_y=2, pressure=3)
    assigned, occupied, next_slot = [], set(), 4
    for field in fields:
        semantic = field.get('semantic', field['name'])
        slot = known.get(semantic)
        if slot is None or slot in occupied:
            slot, next_slot = next_slot, next_slot + 1
        occupied.add(slot)
        assigned.append(slot)
    return assigned


def layout(slots, height, width, device, batch):
    rows, cols, indices = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device),
                                       torch.arange(2 * len(slots), device=device), indexing='ij')
    identities = torch.tensor([2 * slot + family for slot in slots for family in (0, 1)], device=device)
    return tuple(values.reshape(1, -1).expand(batch, -1) for values in (rows, cols, identities[indices]))


def flatten_codes(codes):
    if codes.ndim != 5 or codes.shape[2] != 2:
        raise ValueError('Require batch/field/family/row/column codes')
    codes = codes.long().clone()
    codes[:, :, 1] += MORPHOLOGY
    return codes.permute(0, 3, 4, 1, 2).reshape(len(codes), -1)


def unflatten_codes(sequence, fields, height, width):
    values = sequence.reshape(len(sequence), height, width, fields, 2).permute(0, 3, 4, 1, 2).clone()
    values[:, :, 1] -= MORPHOLOGY
    return values


def token_loss(logits, target):
    morphology = functional.cross_entropy(logits[:, ::2, :MORPHOLOGY].float().reshape(-1, MORPHOLOGY),
                                          target[:, ::2].reshape(-1))
    amplitude = functional.cross_entropy(logits[:, 1::2, MORPHOLOGY:MORPHOLOGY + AMPLITUDE].float().reshape(-1, AMPLITUDE),
                                         (target[:, 1::2] - MORPHOLOGY).reshape(-1))
    return (morphology + amplitude) / 2


class ForecastModel(EncDecSharedGPT):
    def __init__(self, config, statistics, times, architecture=None):
        architecture = ARCHITECTURE if architecture is None else architecture
        fields = config['dynamic_fields'] + config.get('static_fields', [])
        slots = field_slots(fields)
        height, width = [size // 8 for size in config['shape']]
        super().__init__(vocab_size=VOCABULARY, n_embed=MORPHOLOGY + AMPLITUDE, Hq=height, Wq=width,
                         C=max(8, 2 * (max(slots) + 1)),
                         deck_len=2 + (0 if config['conditioning']['encoding'] == 'none' else len(config['parameters'])),
                         dec_block_size=height * width * 2 * len(config['dynamic_fields']),
                         dropout=config['training'].get('dropout', .1), pad_id=PAD, use_sdpa=True, **architecture)
        self.input_slots, self.output_slots = slots, slots[:len(config['dynamic_fields'])]
        self.prefix = ConditioningPrefix(statistics, architecture['n_embd'], times[-1] - times[0],
                                         **config['conditioning'])
        self.gradient_checkpointing = True

    def encode_inputs(self, tokens, parameters, source_time, target_time):
        rows, cols, fields = layout(self.input_slots, self.Hq, self.Wq, tokens.device, len(tokens))
        if tokens.shape != rows.shape:
            raise ValueError('Wrong source token geometry')
        prefix = self.prefix(parameters, source_time, target_time) + self.deck_type
        state = self.drop(torch.cat((prefix, self.tok_emb(tokens) + self.row_emb(rows) + self.col_emb(cols)
                                     + self.channel_emb(fields) + self.frame_type), dim=1))
        zeros = torch.zeros((len(tokens), prefix.shape[1]), device=tokens.device, dtype=torch.long)
        memory_rows, memory_cols = torch.cat((zeros, rows), 1), torch.cat((zeros, cols), 1)
        self.memory_rows, self.memory_cols = memory_rows, memory_cols
        self.memory_padding_mask = torch.zeros(memory_rows.shape, device=tokens.device, dtype=torch.bool)
        for block in self.enc_blocks:
            if self.training and self.gradient_checkpointing:
                state = checkpoint(block, state, memory_rows, memory_cols, use_reentrant=False)
            else:
                state = block(state, memory_rows, memory_cols)
        return self.enc_ln(state)

    def forward(self, source, parameters, source_time, target_time, target):
        memory = self.encode_inputs(source, parameters, source_time, target_time)
        rows, cols, fields = layout(self.output_slots, self.Hq, self.Wq, source.device, len(source))
        if target.shape != rows.shape:
            raise ValueError('Wrong target geometry')
        shifted = torch.cat((torch.full_like(target[:, :1], BOS), target[:, :-1]), dim=1)
        state = self.drop(self.tok_emb(shifted) + self.row_emb(rows) + self.col_emb(cols) + self.channel_emb(fields))
        for block in self.dec_blocks:
            if self.training and self.gradient_checkpointing:
                state = checkpoint(block, state, rows, cols, memory, self.memory_rows, self.memory_cols, use_reentrant=False)
            else:
                state = block(state, rows, cols, memory, self.memory_rows, self.memory_cols)
        return self.lm_head(self.dec_ln(state))

    @torch.inference_mode()
    def generate(self, source, parameters, source_time, target_time, limit=None):
        if self.training:
            raise ValueError('Generation requires eval mode')
        memory = self.encode_inputs(source, parameters, source_time, target_time)
        rows, cols, fields = layout(self.output_slots, self.Hq, self.Wq, source.device, len(source))
        decoder = CachedDecoder(self, memory, rows, cols, fields)
        token = torch.full((len(source), 1), BOS, device=source.device, dtype=torch.long)
        generated = []
        for position in range(rows.shape[1] if limit is None else min(limit, rows.shape[1])):
            logits = decoder.step(token)
            lower, upper = (0, MORPHOLOGY) if position % 2 == 0 else (MORPHOLOGY, MORPHOLOGY + AMPLITUDE)
            token = logits[:, lower:upper].argmax(-1, keepdim=True) + lower
            generated.append(token)
        return torch.cat(generated, 1)


def migrate_pretrained(model, source):
    if (source.get('step') != 140000 or source.get('n_embed') != 9664 or source.get('vocab_size') != VOCABULARY
            or source.get('Hq') != 16 or source.get('Wq') != 16 or source.get('C') != 8
            or source.get('model_type') == 'parallel_encdec_vq'
            or source.get('delta_t_choices') != list(range(1, 21))
            or any(source['cfg'].get(key) != value for key, value in ARCHITECTURE.items())):
        raise ValueError('Require the original AR583 step140k df8 source checkpoint')
    weights = dict(source['model'])
    changes = {}
    for name, size in [('row_emb.weight', model.Hq), ('col_emb.weight', model.Wq)]:
        if size != weights[name].shape[0]:
            weights[name] = functional.interpolate(weights[name].T[None].float(), size=size, mode='linear',
                                                    align_corners=True)[0].T.to(weights[name].dtype)
            changes[name] = dict(old=16, new=size, method='linear_align_corners')
    original = weights['channel_emb.weight']
    if model.C > len(original):
        extended = original.new_empty((model.C, original.shape[1]))
        extended[:len(original)] = original
        for index in range(len(original), model.C):
            extended[index] = original[index % 2::2].mean(0)
        weights['channel_emb.weight'] = extended
        changes['channel_emb.weight'] = dict(old=8, new=model.C, method='family_mean')
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected or set(missing) != {key for key in model.state_dict() if key.startswith('prefix.')}:
        raise ValueError(f'Unexplained checkpoint migration: {missing}, {unexpected}')
    return dict(source_step=140000, changes=changes, new_parameters=missing, token_ids_unchanged=True)

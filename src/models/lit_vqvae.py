import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
from copy import deepcopy

from src.modules.diffusionmodules.model import Encoder, Decoder
from src.modules.vqvae.quantize import VectorQuantizer2
from src.data.transforms import m11_to_physical_torch, physical_to_m11_torch


class LitVQVAE(L.LightningModule):
    """
    Minimal VQ-VAE with optional field conditioning, output transforms,
    and masked physical-space auxiliary loss scaffolding.
    """

    def __init__(
        self,
        ddconfig: dict,
        n_embed: int,
        embed_dim: int,
        learning_rate: float = 2e-4,
        beta: float = 0.25,
        vq_loss_weight: float = 1.0,
        recon_l2_weight: float = 0.0,
        grad_loss_weight: float = 0.0,
        l2_normalize_codebook: bool = False,
        legacy_beta_bug: bool = False,
        output_clamp: bool = True,
        image_key: str = "image",
        usage_entropy_weight: float = 0.03,
        dead_code_reset_every: int = 1000,
        dead_code_reset_threshold: float = 1.0,
        dead_code_reset_warmup_steps: int = 2000,
        max_dead_code_resets: int = 512,
        no_quant: bool = False,
        warmup_steps: int = 0,
        total_steps: int = 0,
        min_lr: float = 1e-6,
        conditioning_mode: str = "none",
        num_field_types: int = 1,
        conditioning_dim: int = 0,
        default_field_id: int = 0,
        use_output_transform: bool = False,
        output_transform_default: str = "identity",
        field_output_transforms: dict | None = None,
        output_transform_eps: float = 1e-6,
        lambda_integral: float = 0.0,
        integral_loss_mode: str = "relative",
        integral_min_target_abs: float = 1e-8,
        use_exact_integral_correction: bool = False,
        exact_integral_mode_default: str = "multiplicative",
        lambda_range: float = 0.0,
        range_loss_mode: str = "hinge",
    ):
        super().__init__()

        ddconfig = deepcopy(ddconfig)
        for k in ("ch_mult", "attn_resolutions"):
            if k in ddconfig and ddconfig[k] is not None:
                ddconfig[k] = list(ddconfig[k])

        self.save_hyperparameters()

        self.image_key = image_key
        self.learning_rate = float(learning_rate)

        in_channels = int(ddconfig["in_channels"])
        out_ch = int(ddconfig["out_ch"])
        z_channels = int(ddconfig["z_channels"])

        self.conditioning_mode = str(conditioning_mode)
        if self.conditioning_mode not in ("none", "concat"):
            raise ValueError(f"Unsupported conditioning_mode={self.conditioning_mode}")
        self.num_field_types = int(max(1, num_field_types))
        self.conditioning_dim = int(max(0, conditioning_dim))
        self.default_field_id = int(default_field_id)
        self.use_output_transform = bool(use_output_transform)
        self.output_transform_default = str(output_transform_default)
        self.field_output_transforms = {} if field_output_transforms is None else dict(field_output_transforms)
        self.output_transform_eps = float(output_transform_eps)
        self.lambda_integral = float(lambda_integral)
        self.integral_loss_mode = str(integral_loss_mode)
        self.integral_min_target_abs = float(integral_min_target_abs)
        self.use_exact_integral_correction = bool(use_exact_integral_correction)
        self.exact_integral_mode_default = str(exact_integral_mode_default)
        self.lambda_range = float(lambda_range)
        self.range_loss_mode = str(range_loss_mode)

        self.use_concat_conditioning = (
            self.conditioning_mode == "concat" and self.conditioning_dim > 0
        )

        encoder_ddconfig = deepcopy(ddconfig)
        decoder_ddconfig = deepcopy(ddconfig)

        if self.use_concat_conditioning:
            encoder_ddconfig["in_channels"] = in_channels + self.conditioning_dim
            decoder_ddconfig["z_channels"] = z_channels + self.conditioning_dim
            self.field_embedding = nn.Embedding(self.num_field_types, self.conditioning_dim)
        else:
            self.field_embedding = None

        self.encoder = Encoder(**encoder_ddconfig)
        self.decoder = Decoder(**decoder_ddconfig)

        self.quant_conv = nn.Conv2d(z_channels, embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, kernel_size=1)

        self.quantize = VectorQuantizer2(
            n_e=n_embed,
            e_dim=embed_dim,
            beta=beta,
            remap=None,
            sane_index_shape=False,
            legacy=legacy_beta_bug,
            l2_normalize=l2_normalize_codebook,
        )

        self.vq_loss_weight = float(vq_loss_weight)
        self.recon_l2_weight = float(recon_l2_weight)
        self.grad_loss_weight = float(grad_loss_weight)
        self.output_clamp = bool(output_clamp)

        self.usage_entropy_weight = float(usage_entropy_weight)
        self.dead_code_reset_every = int(dead_code_reset_every)
        self.dead_code_reset_threshold = float(dead_code_reset_threshold)
        self.dead_code_reset_warmup_steps = int(dead_code_reset_warmup_steps)
        self.max_dead_code_resets = int(max_dead_code_resets)

        self.no_quant = bool(no_quant)

        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        self._base_in_channels = in_channels
        self._base_out_channels = out_ch
        self._base_z_channels = z_channels

    def get_input(self, batch):
        x = batch[self.image_key]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.float()

    def _coerce_field_id(self, x: torch.Tensor, field_id=None):
        B = int(x.shape[0])
        device = x.device
        if field_id is None:
            return torch.full((B,), self.default_field_id, device=device, dtype=torch.long)
        if not torch.is_tensor(field_id):
            field_id = torch.as_tensor(field_id, device=device, dtype=torch.long)
        else:
            field_id = field_id.to(device=device, dtype=torch.long)
        if field_id.ndim == 0:
            field_id = field_id.view(1).expand(B)
        elif field_id.ndim > 1:
            field_id = field_id.reshape(B)
        if field_id.numel() != B:
            if field_id.numel() == 1:
                field_id = field_id.expand(B)
            else:
                raise ValueError(f"field_id has {field_id.numel()} entries but batch size is {B}")
        return field_id.clamp(0, self.num_field_types - 1)

    def _field_map(self, ref: torch.Tensor, field_id=None):
        if not self.use_concat_conditioning:
            return None
        field_id = self._coerce_field_id(ref, field_id=field_id)
        emb = self.field_embedding(field_id)
        return emb[:, :, None, None].expand(-1, -1, ref.shape[-2], ref.shape[-1])

    def _resolve_output_transform_name(self, field_id=None, metadata=None) -> str:
        if metadata is not None and isinstance(metadata, dict):
            cc = metadata.get("constraint_class", None)
            if isinstance(cc, str) and cc:
                if cc in ("identity", "softplus", "sigmoid"):
                    return cc

        if field_id is not None:
            try:
                fid = int(torch.as_tensor(field_id).reshape(-1)[0].item())
                if fid in self.field_output_transforms:
                    return str(self.field_output_transforms[fid])
                if str(fid) in self.field_output_transforms:
                    return str(self.field_output_transforms[str(fid)])
            except Exception:
                pass

        return self.output_transform_default

    def apply_output_transform(self, raw: torch.Tensor, field_id=None, metadata=None) -> torch.Tensor:
        if not self.use_output_transform:
            return raw

        name = self._resolve_output_transform_name(field_id=field_id, metadata=metadata)
        if name == "identity":
            return raw
        if name == "softplus":
            return F.softplus(raw) + self.output_transform_eps
        if name == "sigmoid":
            return torch.sigmoid(raw)
        raise ValueError(f"Unsupported output transform: {name}")
    def apply_exact_integral_correction(self, x_rec: torch.Tensor, x_target: torch.Tensor, metadata=None) -> torch.Tensor:
        if not self.use_exact_integral_correction or not isinstance(metadata, dict):
            return x_rec

        B = int(x_rec.shape[0])
        use_exact = metadata.get("use_exact_integral_correction", False)
        use_exact = torch.as_tensor(use_exact, device=x_rec.device, dtype=torch.bool).reshape(-1)
        if use_exact.numel() == 1:
            use_exact = use_exact.expand(B)
        if not torch.any(use_exact):
            return x_rec

        stats = self._extract_scalar_physical_stats(metadata, B, x_rec.device, x_rec.dtype)
        if stats is None:
            return x_rec
        vmin_t, _vmax_t, vrange_t = stats

        exact_mode = metadata.get("exact_integral_mode", self.exact_integral_mode_default)
        if isinstance(exact_mode, (list, tuple)):
            mode = str(exact_mode[0])
        elif torch.is_tensor(exact_mode):
            mode = str(exact_mode.reshape(-1)[0].item())
        else:
            mode = str(exact_mode)

        x_rec_phys = m11_to_physical_torch(x_rec, vmin_t[:, None, None, None], vrange_t[:, None, None, None])
        x_tgt_phys = m11_to_physical_torch(x_target, vmin_t[:, None, None, None], vrange_t[:, None, None, None])

        pred_sum = x_rec_phys.sum(dim=(1, 2, 3), keepdim=True)
        tgt_sum = x_tgt_phys.sum(dim=(1, 2, 3), keepdim=True)
        x_corr_phys = x_rec_phys.clone()

        if mode == "additive":
            n = float(x_rec_phys.shape[-2] * x_rec_phys.shape[-1])
            delta = (tgt_sum - pred_sum) / max(n, 1.0)
            x_corr_phys[use_exact] = x_rec_phys[use_exact] + delta[use_exact]
        elif mode == "multiplicative":
            scale = tgt_sum / pred_sum.clamp_min(self.integral_min_target_abs)
            x_corr_phys[use_exact] = x_rec_phys[use_exact] * scale[use_exact]
        else:
            return x_rec

        return physical_to_m11_torch(x_corr_phys, vmin_t[:, None, None, None], vrange_t[:, None, None, None])


    def _extract_scalar_physical_stats(self, metadata, batch_size: int, device, dtype):
        if not isinstance(metadata, dict):
            return None
        if "vmin_physical" not in metadata or "vrange_physical" not in metadata:
            return None
        vmin = metadata["vmin_physical"]
        vrange = metadata["vrange_physical"]
        vmax = metadata.get("vmax_physical", None)

        vmin_t = torch.as_tensor(vmin, device=device, dtype=dtype).reshape(-1)
        vrange_t = torch.as_tensor(vrange, device=device, dtype=dtype).reshape(-1)
        if vmax is None:
            vmax_t = vmin_t + vrange_t
        else:
            vmax_t = torch.as_tensor(vmax, device=device, dtype=dtype).reshape(-1)

        if vmin_t.numel() == 1:
            vmin_t = vmin_t.expand(batch_size)
            vrange_t = vrange_t.expand(batch_size)
            vmax_t = vmax_t.expand(batch_size)
        elif vmin_t.numel() != batch_size:
            return None
        return vmin_t, vmax_t, vrange_t

    def _integral_loss_mask(self, metadata, batch_size: int, device):
        if not isinstance(metadata, dict):
            return None
        if "apply_integral_loss" in metadata:
            mask = metadata["apply_integral_loss"]
            mask = torch.as_tensor(mask, device=device, dtype=torch.bool).reshape(-1)
            if mask.numel() == 1:
                mask = mask.expand(batch_size)
            return mask
        cc = metadata.get("constraint_class", None)
        if cc is None:
            return None
        if isinstance(cc, str):
            val = cc in ("positive", "positive_conserved", "conserved_softplus")
            return torch.full((batch_size,), val, device=device, dtype=torch.bool)
        if isinstance(cc, (list, tuple)):
            vals = [str(x) in ("positive", "positive_conserved", "conserved_softplus") for x in cc]
            mask = torch.as_tensor(vals, device=device, dtype=torch.bool)
            if mask.numel() == 1:
                mask = mask.expand(batch_size)
            return mask
        return None

    def compute_integral_loss_physical(self, x_rec: torch.Tensor, x: torch.Tensor, metadata=None):
        B = int(x.shape[0])
        stats = self._extract_scalar_physical_stats(metadata, B, x.device, x.dtype)
        if stats is None:
            return None, None
        vmin_t, _vmax_t, vrange_t = stats
        mask = self._integral_loss_mask(metadata, B, x.device)
        if mask is None:
            return None, None

        x_phys = m11_to_physical_torch(x, vmin_t[:, None, None, None], vrange_t[:, None, None, None])
        x_rec_phys = m11_to_physical_torch(x_rec, vmin_t[:, None, None, None], vrange_t[:, None, None, None])

        target = x_phys.sum(dim=(1, 2, 3))
        pred = x_rec_phys.sum(dim=(1, 2, 3))
        err = (pred - target).abs()

        aux = {
            "integral_active_frac": mask.float().mean(),
            "integral_target_abs_mean": target.abs()[mask].mean() if torch.any(mask) else None,
        }

        if self.integral_loss_mode == "relative":
            valid = mask & (target.abs() >= self.integral_min_target_abs)
            aux["integral_valid_frac"] = valid.float().mean()
            if not torch.any(valid):
                return None, aux
            val = (err[valid] / target[valid].abs().clamp_min(self.integral_min_target_abs)).mean()
            return val, aux

        if self.integral_loss_mode == "absolute":
            valid = mask
            aux["integral_valid_frac"] = valid.float().mean()
            if not torch.any(valid):
                return None, aux
            return err[valid].mean(), aux

        raise ValueError(f"Unsupported integral_loss_mode={self.integral_loss_mode}")

    def _range_loss_mask(self, metadata, batch_size: int, device):
        if not isinstance(metadata, dict):
            return None
        if "apply_range_loss" in metadata:
            mask = metadata["apply_range_loss"]
            mask = torch.as_tensor(mask, device=device, dtype=torch.bool).reshape(-1)
            if mask.numel() == 1:
                mask = mask.expand(batch_size)
            return mask
        cc = metadata.get("constraint_class", None)
        if cc is None:
            return None
        valid_classes = ("positive", "positive_conserved", "conserved_softplus", "bounded")
        if isinstance(cc, str):
            return torch.full((batch_size,), cc in valid_classes, device=device, dtype=torch.bool)
        if isinstance(cc, (list, tuple)):
            vals = [str(x) in valid_classes for x in cc]
            mask = torch.as_tensor(vals, device=device, dtype=torch.bool)
            if mask.numel() == 1:
                mask = mask.expand(batch_size)
            return mask
        return None

    def compute_range_loss_physical(self, x_rec: torch.Tensor, metadata=None):
        B = int(x_rec.shape[0])
        stats = self._extract_scalar_physical_stats(metadata, B, x_rec.device, x_rec.dtype)
        if stats is None:
            return None, None
        vmin_t, vmax_t, vrange_t = stats
        mask = self._range_loss_mask(metadata, B, x_rec.device)
        if mask is None or not torch.any(mask):
            return None, None

        x_rec_phys = m11_to_physical_torch(x_rec, vmin_t[:, None, None, None], vrange_t[:, None, None, None])
        lower = vmin_t[:, None, None, None]
        upper = vmax_t[:, None, None, None]

        below = (lower - x_rec_phys).clamp_min(0.0)
        above = (x_rec_phys - upper).clamp_min(0.0)

        if self.range_loss_mode == "hinge":
            per_sample = (below + above).mean(dim=(1, 2, 3))
        elif self.range_loss_mode == "mse":
            per_sample = (below.square() + above.square()).mean(dim=(1, 2, 3))
        else:
            raise ValueError(f"Unsupported range_loss_mode={self.range_loss_mode}")

        val = per_sample[mask].mean()
        aux = {"range_active_frac": mask.float().mean()}
        return val, aux


    def encode_prequant(self, x, field_id=None, metadata=None):
        if self.use_concat_conditioning:
            cond = self._field_map(x, field_id=field_id)
            x = torch.cat([x, cond], dim=1)
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def encode(self, x, field_id=None, metadata=None):
        h = self.encode_prequant(x, field_id=field_id, metadata=metadata)

        if self.no_quant:
            qloss = torch.zeros((), device=h.device, dtype=h.dtype)
            info = (None, None, None)
            return h, qloss, info

        z_q, qloss, info = self.quantize(h)
        return z_q, qloss, info

    def decode(self, z_q, field_id=None, metadata=None):
        z_q = self.post_quant_conv(z_q)
        if self.use_concat_conditioning:
            cond = self._field_map(z_q, field_id=field_id)
            z_q = torch.cat([z_q, cond], dim=1)
        raw = self.decoder(z_q)
        x_rec = self.apply_output_transform(raw, field_id=field_id, metadata=metadata)
        return x_rec

    def forward(self, x, field_id=None, metadata=None):
        z_q, qloss, info = self.encode(x, field_id=field_id, metadata=metadata)
        x_rec = self.decode(z_q, field_id=field_id, metadata=metadata)
        x_rec = self.apply_exact_integral_correction(x_rec, x, metadata=metadata)
        return x_rec, qloss, info

    @staticmethod
    def grad_loss(x_rec, x):
        dx_rec = x_rec[..., :, 1:] - x_rec[..., :, :-1]
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy_rec = x_rec[..., 1:, :] - x_rec[..., :-1, :]
        dy = x[..., 1:, :] - x[..., :-1, :]
        return F.l1_loss(dx_rec, dx) + F.l1_loss(dy_rec, dy)

    @staticmethod
    def codebook_usage_entropy_loss(inds: torch.Tensor, n_embed: int, eps: float = 1e-8):
        counts = torch.bincount(inds, minlength=n_embed).float()
        p = counts / (counts.sum() + eps)
        entropy = -(p * (p + eps).log()).sum()
        return -entropy, entropy

    def _ddp_global_counts(self, inds_1d: torch.Tensor) -> torch.Tensor:
        n_embed = self.quantize.n_e
        if self.trainer is None or self.trainer.world_size == 1:
            return torch.bincount(inds_1d, minlength=n_embed).float()
        gathered = self.all_gather(inds_1d)
        all_inds = gathered.reshape(-1)
        return torch.bincount(all_inds, minlength=n_embed).float()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch)
        field_id = batch.get("field_id", None) if isinstance(batch, dict) else None
        metadata = batch if isinstance(batch, dict) else None

        h = self.encode_prequant(x, field_id=field_id, metadata=metadata)

        if self.no_quant:
            z_q = h
            qloss = torch.zeros((), device=h.device, dtype=h.dtype)
            info = (None, None, None)
        else:
            z_q, qloss, info = self.quantize(h)

        x_rec = self.decode(z_q, field_id=field_id, metadata=metadata)

        if self.output_clamp:
            x_rec = x_rec.clamp(-1, 1)

        recon_l1 = F.l1_loss(x_rec, x)
        loss = recon_l1

        recon_l2 = None
        if self.recon_l2_weight > 0:
            recon_l2 = F.mse_loss(x_rec, x)
            loss = loss + self.recon_l2_weight * recon_l2

        loss = loss + self.vq_loss_weight * qloss

        gl = None
        if self.grad_loss_weight > 0:
            gl = self.grad_loss(x_rec, x)
            loss = loss + self.grad_loss_weight * gl

        integral_loss = None
        integral_aux = None
        if self.lambda_integral > 0:
            integral_loss, integral_aux = self.compute_integral_loss_physical(x_rec, x, metadata=metadata)
            if integral_loss is not None:
                loss = loss + self.lambda_integral * integral_loss

        range_loss = None
        range_aux = None
        if self.lambda_range > 0:
            range_loss, range_aux = self.compute_range_loss_physical(x_rec, metadata=metadata)
            if range_loss is not None:
                loss = loss + self.lambda_range * range_loss

        if (not self.no_quant) and (info is not None) and (len(info) > 0) and (info[-1] is not None):
            inds = info[-1].reshape(-1).long()
            counts = self._ddp_global_counts(inds)
            p = counts / (counts.sum() + 1e-8)
            entropy = -(p * (p + 1e-8).log()).sum()
            perplexity = torch.exp(entropy)
            unique_global = (counts > 0).sum()

            if self.usage_entropy_weight > 0:
                loss = loss - self.usage_entropy_weight * entropy
        else:
            counts = None
            entropy = None
            perplexity = None
            unique_global = None

        if (
            counts is not None
            and self.dead_code_reset_every > 0
            and self.global_step >= self.dead_code_reset_warmup_steps
            and (self.global_step % self.dead_code_reset_every == 0)
        ):
            dead = (counts <= self.dead_code_reset_threshold).nonzero(as_tuple=False).squeeze(1)
            if dead.numel() > 0:
                if self.max_dead_code_resets > 0 and dead.numel() > self.max_dead_code_resets:
                    perm = torch.randperm(dead.numel(), device=dead.device)
                    dead = dead[perm[: self.max_dead_code_resets]]
                h_flat = h.detach().permute(0, 2, 3, 1).reshape(-1, h.shape[1])
                if h_flat.shape[0] > 0:
                    repl_idx = torch.randint(0, h_flat.shape[0], (dead.numel(),), device=h_flat.device)
                    new_vecs = h_flat[repl_idx]
                    with torch.no_grad():
                        self.quantize.embedding.weight.data[dead] = new_vecs.to(self.quantize.embedding.weight.dtype)
                    self.log("train/dead_codes_reset", float(dead.numel()), prog_bar=False, on_step=True, on_epoch=False)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/recon_l1", recon_l1, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/qloss", qloss, prog_bar=False, on_step=True, on_epoch=True)
        if recon_l2 is not None:
            self.log("train/recon_l2", recon_l2, prog_bar=False, on_step=True, on_epoch=True)
        if gl is not None:
            self.log("train/grad_l1", gl, prog_bar=False, on_step=True, on_epoch=True)
        if integral_loss is not None:
            self.log("train/integral_loss_physical", integral_loss, prog_bar=False, on_step=True, on_epoch=True)
        if range_loss is not None:
            self.log("train/range_loss_physical", range_loss, prog_bar=False, on_step=True, on_epoch=True)
        if integral_aux is not None:
            if integral_aux.get("integral_active_frac", None) is not None:
                self.log("train/integral_active_frac", integral_aux["integral_active_frac"], prog_bar=False, on_step=True, on_epoch=True)
            if integral_aux.get("integral_valid_frac", None) is not None:
                self.log("train/integral_valid_frac", integral_aux["integral_valid_frac"], prog_bar=False, on_step=True, on_epoch=True)
            if integral_aux.get("integral_target_abs_mean", None) is not None:
                self.log("train/integral_target_abs_mean", integral_aux["integral_target_abs_mean"], prog_bar=False, on_step=True, on_epoch=True)
        if range_aux is not None and range_aux.get("range_active_frac", None) is not None:
            self.log("train/range_active_frac", range_aux["range_active_frac"], prog_bar=False, on_step=True, on_epoch=True)
        if entropy is not None:
            self.log("train/code_entropy", entropy, prog_bar=False, on_step=True, on_epoch=True)
        if perplexity is not None:
            self.log("train/code_perplexity", perplexity, prog_bar=False, on_step=True, on_epoch=True)
        if unique_global is not None:
            self.log("train/code_unique_global", unique_global.float(), prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        field_id = batch.get("field_id", None) if isinstance(batch, dict) else None
        metadata = batch if isinstance(batch, dict) else None
        x_rec, qloss, info = self(x, field_id=field_id, metadata=metadata)

        if self.output_clamp:
            x_rec = x_rec.clamp(-1, 1)

        recon_l1 = F.l1_loss(x_rec, x)
        loss = recon_l1 + self.vq_loss_weight * qloss

        recon_l2 = None
        if self.recon_l2_weight > 0:
            recon_l2 = F.mse_loss(x_rec, x)
            loss = loss + self.recon_l2_weight * recon_l2

        gl = None
        if self.grad_loss_weight > 0:
            gl = self.grad_loss(x_rec, x)
            loss = loss + self.grad_loss_weight * gl

        integral_loss = None
        integral_aux = None
        if self.lambda_integral > 0:
            integral_loss, integral_aux = self.compute_integral_loss_physical(x_rec, x, metadata=metadata)
            if integral_loss is not None:
                loss = loss + self.lambda_integral * integral_loss

        range_loss = None
        range_aux = None
        if self.lambda_range > 0:
            range_loss, range_aux = self.compute_range_loss_physical(x_rec, metadata=metadata)
            if range_loss is not None:
                loss = loss + self.lambda_range * range_loss

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/recon_l1", recon_l1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/qloss", qloss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        if recon_l2 is not None:
            self.log("val/recon_l2", recon_l2, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        if gl is not None:
            self.log("val/grad_l1", gl, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        if integral_loss is not None:
            self.log("val/integral_loss_physical", integral_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        if range_loss is not None:
            self.log("val/range_loss_physical", range_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        if integral_aux is not None:
            if integral_aux.get("integral_active_frac", None) is not None:
                self.log("val/integral_active_frac", integral_aux["integral_active_frac"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            if integral_aux.get("integral_valid_frac", None) is not None:
                self.log("val/integral_valid_frac", integral_aux["integral_valid_frac"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            if integral_aux.get("integral_target_abs_mean", None) is not None:
                self.log("val/integral_target_abs_mean", integral_aux["integral_target_abs_mean"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        if range_aux is not None and range_aux.get("range_active_frac", None) is not None:
            self.log("val/range_active_frac", range_aux["range_active_frac"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return {"val_loss": loss.detach(), "val_recon_l1": recon_l1.detach()}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.95), weight_decay=1e-4)
        if self.total_steps <= 0:
            return opt

        def lr_lambda(step: int):
            if self.warmup_steps > 0 and step < self.warmup_steps:
                return float(step + 1) / float(max(1, self.warmup_steps))
            if self.total_steps <= self.warmup_steps:
                return 1.0
            progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = self.min_lr / max(self.learning_rate, 1e-12)
            return min_ratio + (1.0 - min_ratio) * cosine

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }

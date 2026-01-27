import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .utils import hash_state_dict_keys, init_weights_on_device

# try:
#     import flash_attn_interface
#     FLASH_ATTN_3_AVAILABLE = True
# except ModuleNotFoundError:
#     FLASH_ATTN_3_AVAILABLE = False
# try:
#     import flash_attn
#     FLASH_ATTN_2_AVAILABLE = True
# except ModuleNotFoundError:
#     FLASH_ATTN_2_AVAILABLE = False
# try:
#     from sageattention import sageattn
#     SAGE_ATTN_AVAILABLE = True
# except ModuleNotFoundError:
#     SAGE_ATTN_AVAILABLE = False


# def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
#     if compatibility_mode:
#         q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
#         x = F.scaled_dot_product_attention(q, k, v)
#         x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
#     elif FLASH_ATTN_3_AVAILABLE:
#         q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
#         x = flash_attn_interface.flash_attn_func(q, k, v)
#         if isinstance(x, tuple):
#             x = x[0]
#         x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
#     elif FLASH_ATTN_2_AVAILABLE:
#         q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
#         x = flash_attn.flash_attn_func(q, k, v)
#         x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
#     elif SAGE_ATTN_AVAILABLE:
#         q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
#         x = sageattn(q, k, v)
#         x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
#     else:
#         q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
#         x = F.scaled_dot_product_attention(q, k, v)
#         x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
#     return x


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, attn_mask: Optional[torch.Tensor] = None):
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0, delta: Optional[int] = None):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    if delta is None:
        freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    else:
        freqs = torch.outer(torch.arange(end, device=freqs.device) + delta, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, attn_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, attn_mask=attn_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, ref, ref_freqs, save_dir=None, step_idx=None, block_idx=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)  # [1, 75600, 5120]
        k = rope_apply(k, freqs, self.num_heads)

        ref_q = self.norm_q(self.q(ref))
        ref_k = self.norm_k(self.k(ref))
        ref_v = self.v(ref)
        ref_q = rope_apply(ref_q, ref_freqs, self.num_heads)
        ref_k = rope_apply(ref_k, ref_freqs, self.num_heads)

        q = torch.cat([q, ref_q], dim=1)  # [1, 75600+3600, 5120]
        k = torch.cat([k, ref_k], dim=1)
        v = torch.cat([v, ref_v], dim=1)

        # mask = torch.ones((q.shape[1], q.shape[1]), device=q.device)
        # mask[:x.shape[1], :] = 0
        # mask_i = x.shape[1]
        # mask_j = x.shape[1] + ref.shape[1]
        # mask[mask_i:mask_j, mask_i:mask_j] = 0
        # mask = mask * -1e20
        # mask = mask.to(dtype=q.dtype)

        if save_dir is not None:
            torch.save(q.detach().cpu(), os.path.join(save_dir, f"step{step_idx}_block{block_idx}_q.pt"))
            torch.save(k.detach().cpu(), os.path.join(save_dir, f"step{step_idx}_block{block_idx}_k.pt"))
            torch.save(v.detach().cpu(), os.path.join(save_dir, f"step{step_idx}_block{block_idx}_v.pt"))

        attn = self.attn(q, k, v)
        x, ref = attn[:, :x.shape[1]], attn[:, x.shape[1]:]

        x = self.o(x)
        ref = self.o(ref)

        return x, ref


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor, save_dir=None, step_idx=None, block_idx=None):
        if self.has_image_input:
            img = y[:, :257]  # [1, 257, 5120]
            ctx = y[:, 257:]  # [1, 512, 5120]
        else:
            ctx = y
        q = self.norm_q(self.q(x))  # [1, 75600, 5120]
        k = self.norm_k(self.k(ctx))  # [1, 512, 5120]
        v = self.v(ctx)  # [1, 512, 5120]

        if save_dir is not None:
            torch.save(q.detach().cpu(), os.path.join(save_dir, f"step{step_idx}_block{block_idx}_q.pt"))
            torch.save(k.detach().cpu(), os.path.join(save_dir, f"step{step_idx}_block{block_idx}_k.pt"))
            torch.save(v.detach().cpu(), os.path.join(save_dir, f"step{step_idx}_block{block_idx}_v.pt"))

        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        ref,
        ref_t_mod,
        ref_freqs,
        sa_save_dir=None,
        ca_save_dir=None,
        step_idx=None,
        block_idx=None,
    ):
        if t_mod.dim() == 3:  # [b 6 dim]
            modulation = self.modulation
            out = (modulation.to(t_mod) + t_mod).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = out  # [b 1 dim]
        else:  # [b 6 n dim]
            modulation = self.modulation.unsqueeze(2)
            out = (modulation.to(t_mod) + t_mod).chunk(6, dim=1)
            out = [o.squeeze(1) for o in out]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = out  # [b n dim]

        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        #     modulation + t_mod
        # ).chunk(6, dim=1)

        ref_modulation = self.modulation
        out = (ref_modulation.to(ref_t_mod) + ref_t_mod).chunk(6, dim=1)
        ref_shift_msa, ref_scale_msa, ref_gate_msa, ref_shift_mlp, ref_scale_mlp, ref_gate_mlp = out

        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        input_ref = modulate(self.norm1(ref), ref_shift_msa, ref_scale_msa)

        output_x, output_ref = self.self_attn(
            input_x,
            freqs,
            input_ref,
            ref_freqs,
            save_dir=sa_save_dir,
            step_idx=step_idx,
            block_idx=block_idx,
        )
        x = self.gate(x, gate_msa, output_x)
        ref = self.gate(ref, ref_gate_msa, output_ref)

        x = x + self.cross_attn(
            self.norm3(x),
            context,
            save_dir=ca_save_dir,
            step_idx=step_idx,
            block_idx=block_idx,
        )

        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        input_ref = modulate(self.norm2(ref), ref_shift_mlp, ref_scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        ref = self.gate(ref, ref_gate_mlp, self.ffn(input_ref))

        return x, ref


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if t_mod.dim() == 2:  # [1, 5120]
            modulation = self.modulation  # [1, 2, 5120]
            shift, scale = (modulation.to(t_mod) + t_mod.unsqueeze(1)).chunk(2, dim=1)
        else:  # [1, fhw, 5120]
            modulation = self.modulation.unsqueeze(2)  # [1, 2, 1, 5120]
            shift, scale = (modulation.to(t_mod) + t_mod.unsqueeze(1)).chunk(2, dim=1)
            shift, scale = shift.squeeze(1), scale.squeeze(1)

        # shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)

        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        shift_ref_pos: bool = False,
        use_hand: bool = False,
        use_hand_proj: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.ref_patch_embedding = nn.Conv3d(16, dim, kernel_size=patch_size, stride=patch_size)
        self.use_hand = use_hand
        self.use_hand_proj = use_hand_proj
        if self.use_hand:
            if self.use_hand_proj:
                self.hand_patch_embedding = nn.Sequential(
                    nn.Conv3d(16, dim, kernel_size=patch_size, stride=patch_size),
                    nn.Conv3d(dim, dim, kernel_size=1),
                )
            else:
                self.hand_patch_embedding = nn.Conv3d(16, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        self.shift_ref_pos = shift_ref_pos
        ref_freqs_dims = (head_dim - 2 * (head_dim // 3), head_dim // 3, head_dim // 3)
        ref_freqs_f = precompute_freqs_cis(ref_freqs_dims[0], 1024)
        ref_freqs_h = precompute_freqs_cis(ref_freqs_dims[1], 1024)
        ref_freqs_w = precompute_freqs_cis(ref_freqs_dims[2], 1024, delta=-1024 if self.shift_ref_pos else None)
        self.ref_freqs = (ref_freqs_f, ref_freqs_h, ref_freqs_w)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        self.has_image_pos_emb = has_image_pos_emb

    def patchify(self, x: torch.Tensor, cond_latents: torch.Tensor, hand_latents: torch.Tensor = None):
        x = torch.cat([x, cond_latents], dim=1)  # [1, 16+20=36, 21, 160, 90]
        x = self.patch_embedding(x)  # [b 36 f h w] -> [b 5120 f h/2 w/2]
        if hand_latents is not None:
            hand_latents = self.hand_patch_embedding(hand_latents)  # [b 16 f h w] -> [b 5120 f h/2 w/2]
            x = x + hand_latents
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def ref_patchify(self, x: torch.Tensor):
        x = self.ref_patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        ref: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        cond_latents: Optional[torch.Tensor] = None,
        hand_latents: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        sa_save_dir=None,
        ca_save_dir=None,
        step_idx=None,
        **kwargs,
    ):
        if timestep.dim() == 1:
            _flag_df = False
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        elif timestep.dim() == 2:
            _flag_df = True
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep.view(-1)))
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        # t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        ref_timestep = torch.zeros((timestep.shape[0],)).to(timestep)
        ref_t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, ref_timestep))
        ref_t_mod = self.time_projection(ref_t).unflatten(1, (6, self.dim))

        context = self.text_embedding(context)  # [1, 512, 4096] -> [1, 512, 5120]

        assert self.has_image_input
        clip_embdding = self.img_emb(clip_feature)  # [1, 257, 1280] -> [1, 257, 5120]
        context = torch.cat([clip_embdding, context], dim=1)

        # x = torch.cat([x, y], dim=1)
        x, (f, h, w) = self.patchify(x, cond_latents, hand_latents)
        ref, (ref_f, ref_h, ref_w) = self.ref_patchify(ref)

        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        freqs = freqs.reshape(f * h * w, 1, -1).to(x.device)

        if not self.shift_ref_pos:
            ref_freqs = torch.cat(
                [
                    self.ref_freqs[0][:ref_f].view(ref_f, 1, 1, -1).expand(ref_f, ref_h, ref_w, -1),
                    self.ref_freqs[1][:ref_h].view(1, ref_h, 1, -1).expand(ref_f, ref_h, ref_w, -1),
                    self.ref_freqs[2][:ref_w].view(1, 1, ref_w, -1).expand(ref_f, ref_h, ref_w, -1),
                ],
                dim=-1,
            )
        else:
            ref_freqs = torch.cat(
                [
                    self.ref_freqs[0][:ref_f].view(ref_f, 1, 1, -1).expand(ref_f, ref_h, ref_w, -1),
                    self.ref_freqs[1][:ref_h].view(1, ref_h, 1, -1).expand(ref_f, ref_h, ref_w, -1),
                    self.ref_freqs[2][-ref_w:].view(1, 1, ref_w, -1).expand(ref_f, ref_h, ref_w, -1),
                ],
                dim=-1,
            )
        ref_freqs = ref_freqs.reshape(ref_f * ref_h * ref_w, 1, -1).to(x.device)

        if _flag_df:
            b = timestep.shape[0]
            t = t.view(b, f, 1, 1, self.dim).repeat(1, 1, h, w, 1).flatten(1, 3)  # [b (f h w) c]
            t_mod = t_mod.view(b, f, 1, 1, 6, self.dim).repeat(1, 1, h, w, 1, 1).flatten(1, 3)  # [b (f h w) 6 c]
            t_mod = t_mod.transpose(1, 2).contiguous()  # [b 6 (f h w) c]

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for block_idx, block in enumerate(self.blocks):
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, ref = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            context,
                            t_mod,
                            freqs,
                            ref,
                            ref_t_mod,
                            ref_freqs,
                            use_reentrant=False,
                        )
                else:
                    x, ref = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        ref,
                        ref_t_mod,
                        ref_freqs,
                        use_reentrant=False,
                    )
            else:
                x, ref = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                    ref,
                    ref_t_mod,
                    ref_freqs,
                    sa_save_dir=sa_save_dir,
                    ca_save_dir=ca_save_dir,
                    step_idx=step_idx,
                    block_idx=block_idx,
                )

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()

    @classmethod
    def from_pretrained(cls, model_path, device="cpu", torch_dtype=torch.bfloat16, **kwargs):
        import os
        import re
        from safetensors.torch import load_file

        pattern = re.compile(r"diffusion_pytorch_model-\d{5}-of-\d{5}\.safetensor")
        sharded_files = [f for f in os.listdir(model_path) if pattern.match(f)]
        sharded_files = [os.path.join(model_path, f) for f in sorted(sharded_files)]
        print(f"[WanModel] Loading pretrained model from: {sharded_files}")

        state_dict = {}
        for file in sharded_files:
            state_dict.update(load_file(file))

        model_state_dict, model_kwargs = cls.state_dict_converter().from_civitai(state_dict)
        if kwargs:
            model_kwargs.update(kwargs)

        pe_state_dict = torch.load("data/weights/patch_embedding.pt", weights_only=True, map_location="cpu")
        if model_kwargs.get("use_hand", False):
            if model_kwargs.get("use_hand_proj", False):
                model_state_dict["hand_patch_embedding.0.weight"] = pe_state_dict["patch_embedding.weight"].clone()
                model_state_dict["hand_patch_embedding.0.bias"] = pe_state_dict["patch_embedding.bias"].clone()
                model_state_dict["hand_patch_embedding.1.weight"] = torch.zeros((model_kwargs["dim"], model_kwargs["dim"], 1, 1, 1))
                model_state_dict["hand_patch_embedding.1.bias"] = torch.zeros((model_kwargs["dim"],))
            else:
                model_state_dict["hand_patch_embedding.weight"] = torch.zeros_like(pe_state_dict["patch_embedding.weight"])
                model_state_dict["hand_patch_embedding.bias"] = torch.zeros_like(pe_state_dict["patch_embedding.bias"])
        model_state_dict["ref_patch_embedding.weight"] = pe_state_dict["patch_embedding.weight"].clone()
        model_state_dict["ref_patch_embedding.bias"] = pe_state_dict["patch_embedding.bias"].clone()
        del pe_state_dict

        with init_weights_on_device():
            model = cls(**model_kwargs)
        model.load_state_dict(model_state_dict, strict=True, assign=True)
        model.to(device=device, dtype=torch_dtype)

        return model, model_kwargs


class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config

    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict, config

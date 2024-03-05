# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Optional, Tuple

import torch
import transformers
from torch import nn


class FusedQK(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(FusedQK, self).__init__(*args, **kwargs)


class OPTGatedAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper
    https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.qk_proj = FusedQK(embed_dim, embed_dim * 2, bias=bias)

        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        no_gating: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query/key proj
        query_key_states = self.qk_proj(hidden_states)

        # get query proj
        query_states = query_key_states[..., : query_key_states.shape[-1] // 2] * self.scaling

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            raise NotImplementedError(f"QK fusion not implmented for cross attention")
        elif is_cross_attention:
            raise NotImplementedError(f"QK fusion not implmented for cross attention")
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = query_key_states[..., query_key_states.shape[-1] // 2 :]

            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = query_key_states[..., query_key_states.shape[-1] // 2 :]

            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                torch.float16
            )
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


def qk_fuse(model):
    # create dict with new layers
    to_replace = {}
    for n, m in model.named_modules():
        if isinstance(m, transformers.models.opt.modeling_opt.OPTAttention):
            embed_dim = m.embed_dim
            num_heads = m.num_heads
            dropout = m.dropout
            is_decoder = m.is_decoder
            bias = m.k_proj.bias is not None
            gated_attention = OPTGatedAttention(embed_dim, num_heads, dropout, is_decoder, bias)

            for old_name, old_param in m.named_parameters():
                found = False

                if "q_proj" in old_name:
                    if "weight" in old_name:
                        print("QK fusion of Q.weight:", old_name, old_param.shape)
                        gated_attention.qk_proj.weight.data[: len(old_param), :] = old_param.data
                        print(gated_attention.qk_proj.weight.data.shape)
                    elif "bias" in old_name:
                        print("QK fusion of Q.bias:", old_name, old_param.shape)
                        gated_attention.qk_proj.bias.data[: len(old_param)] = old_param.data
                        print(gated_attention.qk_proj.bias.data.shape)
                    else:
                        raise ValueError(f"[Err] Where to place: {old_name}")
                    found = True
                elif "k_proj" in old_name:
                    if "weight" in old_name:
                        print("QK fusion of K.weight:", old_name, old_param.shape)
                        gated_attention.qk_proj.weight.data[len(old_param) :, :] = old_param.data
                        print(gated_attention.qk_proj.weight.data.shape)
                    elif "bias" in old_name:
                        print("QK fusion of K.bias:", old_name, old_param.shape)
                        gated_attention.qk_proj.bias.data[len(old_param) :] = old_param.data
                        print(gated_attention.qk_proj.bias.data.shape)
                    else:
                        raise ValueError(f"[Err] Where to place: {old_name}")
                    found = True
                else:
                    for new_name, new_param in gated_attention.named_parameters():
                        if old_name == new_name:
                            found = True
                            new_param.data = old_param.data
                            break

                if not found:
                    raise ValueError(f"No new parameter found with name {old_name}...")

            dtype, device = m.v_proj.weight.dtype, m.v_proj.weight.device

            for param in gated_attention.parameters():
                param.data = param.data.type(dtype)

            to_replace[n] = gated_attention

    # actual replacement
    for n, p in to_replace.items():
        subm = model
        for subn in n.split(".")[:-1]:
            subm = getattr(subm, subn)
        setattr(subm, n.split(".")[-1], p)
        print(f"replaced {n}")

# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Optional, Tuple

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator

from eval import eval_model

import math
from transformers.models.opt.modeling_opt import OPTAttention


class OPTLoraAttention(nn.Module):
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
        lora_dim: int = 8,
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
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.q_lora1 = nn.Linear(embed_dim, lora_dim, bias=False)
        self.k_lora1 = nn.Linear(embed_dim, lora_dim, bias=False)
        self.v_lora1 = nn.Linear(embed_dim, lora_dim, bias=False)
        self.out_lora1 = nn.Linear(embed_dim, lora_dim, bias=False)

        self.q_lora2 = nn.Linear(lora_dim, embed_dim, bias=False)
        self.k_lora2 = nn.Linear(lora_dim, embed_dim, bias=False)
        self.v_lora2 = nn.Linear(lora_dim, embed_dim, bias=False)
        self.out_lora2 = nn.Linear(lora_dim, embed_dim, bias=False)
            
        self.q_lora1.weight.data.normal_(0.0, 1.0)
        self.k_lora1.weight.data.normal_(0.0, 1.0)
        self.v_lora1.weight.data.normal_(0.0, 1.0)
        self.out_lora1.weight.data.normal_(0.0, 1.0)
        
        self.q_lora2.weight.data.zero_()
        self.k_lora2.weight.data.zero_()
        self.v_lora2.weight.data.zero_()
        self.out_lora2.weight.data.zero_()


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

        # get query proj
        row_mask = torch.all(self.q_proj.weight == 0.0, 1)
        col_mask = torch.all(self.q_proj.weight == 0.0, 0)
        self.q_lora1.weight.data[:, col_mask] = 0.0
        self.q_lora2.weight.data[row_mask, :] = 0.0

        query_states = (self.q_proj(hidden_states) + self.q_lora2(self.q_lora1(hidden_states))) * self.scaling

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            raise NotImplementedError(f"Cross attention and past_key_value not implemented")
        elif is_cross_attention:
            raise NotImplementedError(f"Cross attention and past_key_value not implemented")
        elif past_key_value is not None:
            raise NotImplementedError(f"Cross attention and past_key_value not implemented")
        else:
            # self_attention
            row_mask = torch.all(self.k_proj.weight == 0.0, 1)
            col_mask = torch.all(self.k_proj.weight == 0.0, 0)
            self.k_lora1.weight.data[:, col_mask] = 0.0
            self.k_lora2.weight.data[row_mask, :] = 0.0
            key_states = self._shape(self.k_proj(hidden_states) + self.k_lora2(self.k_lora1(hidden_states)), -1, bsz)

            row_mask = torch.all(self.v_proj.weight == 0.0, 1)
            col_mask = torch.all(self.v_proj.weight == 0.0, 0)
            self.v_lora1.weight.data[:, col_mask] = 0.0
            self.v_lora2.weight.data[row_mask, :] = 0.0
            value_states = self._shape(self.v_proj(hidden_states) + self.v_lora2(self.v_lora1(hidden_states)), -1, bsz)

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
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
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

        row_mask = torch.all(self.out_proj.weight == 0.0, 1)
        col_mask = torch.all(self.out_proj.weight == 0.0, 0)
        self.out_lora1.weight.data[:, col_mask] = 0.0
        self.out_lora2.weight.data[row_mask, :] = 0.0

        attn_output = self.out_proj(attn_output) + self.out_lora2(self.out_lora1(attn_output))

        return attn_output, attn_weights_reshaped, past_key_value

def add_lora(model):
    lora_params = []

    # create dict with new layers
    to_replace = {}
    for n, m in model.named_modules():
        if isinstance(m, OPTAttention):
            embed_dim = m.embed_dim
            num_heads = m.num_heads
            dropout = m.dropout
            is_decoder = m.is_decoder
            bias = m.k_proj.bias is not None

            lora_attention = OPTLoraAttention(embed_dim, num_heads, dropout, is_decoder, bias)

            for old_name, old_param in m.named_parameters():
                found = False

                for new_name, new_param in lora_attention.named_parameters():
                    if old_name == new_name:
                        found = True
                        new_param.data = old_param.data
                        break

                if not found:
                    raise ValueError(f"No new parameter found with name {old_name}...")
                
            dtype, device = m.v_proj.weight.dtype, m.v_proj.weight.device

            for param in lora_attention.parameters():
                param.data = param.data
                param = param.to(dtype=dtype, device=device)

            lora_layers = [lora_attention.q_lora1, lora_attention.q_lora2, lora_attention.k_lora1, lora_attention.k_lora2, lora_attention.v_lora1, lora_attention.v_lora2, lora_attention.out_lora1, lora_attention.out_lora2]

            for lora_layer in lora_layers:
                lora_params += list(lora_layer.parameters())

            lora_attention.to(device)
            
            to_replace[n] = lora_attention
            
    # actual replacement
    for n, p in to_replace.items():
        subm = model
        for subn in n.split('.')[:-1]:
            subm = getattr(subm, subn)
        print(f'\treplaced {n}')

    return lora_params

def tune_lora(model, model_str, trainencs, testenc, writer, writer_str, lora_subset, log=False, n_epochs=10, lr=0.0001, dev=None, batch_size=1):
    print('Tuning lora...')

    lora_params = add_lora(model)

    model.float()
    model.eval()

    data_collator = DefaultDataCollator

    indices = np.arange(len(trainencs))
    np.random.shuffle(indices)
    indices = list(indices[:int(len(indices) * lora_subset)])

    train_dataloader = DataLoader(
        [x for i, x in enumerate(trainencs) if i in indices], shuffle=False, batch_size=1, num_workers=4, collate_fn=data_collator
    )

    for param in model.parameters():
        param.requires_grad = False
        
    for param in lora_params:
        param.requires_grad = True

    optimizer = optim.Adam(lora_params, lr=lr)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=n_epochs)

    for epoch in range(n_epochs):
        model.to(dev)
        
        losses = 0.0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            if len(batch.return_tensors[0]) == 2:
                batch = torch.tensor(batch.return_tensors[0][0]).to(dev)
            elif len(batch.return_tensors[0]) > 2:
                batch = torch.tensor([batch.return_tensors[0]]).to(dev)
            else:
                raise ValueError(f"Something went wrong parsing the input batch shape")

            out = model(batch)
            logits = out['logits']
            
            loss = nn.CrossEntropyLoss()
            
            L = loss(logits[:, :-1, :].view(-1, logits.size(-1)), batch[:, 1:].view(-1))
            
            L.backward()
            
            optimizer.step()
            
            losses += L.item()
            
        scheduler.step()
        losses = losses / len(train_dataloader)

        print("Done LoRA: ", math.exp(losses))
        
        print(f"Evaluating...")
        test_outdir = eval_model(model, model_str, testenc)
        print(f'Test  PPL [en]:', test_outdir['ppl'])

        if writer is not None:
            writer.add_scalar(f'lora/{writer_str}_ppl', test_outdir['ppl'], epoch)


def undo_lora(model):
    print('Undoing lora terms...')

    # create dict with new layers
    to_replace = {}
    for n, m in model.named_modules():
        if isinstance(m, OPTLoraAttention):
            embed_dim = m.embed_dim
            num_heads = m.num_heads
            dropout = m.dropout
            is_decoder = m.is_decoder
            bias = m.k_proj.bias is not None

            attention = OPTAttention(embed_dim, num_heads, dropout, is_decoder, bias)

            q_lora = m.q_lora2.weight @ m.q_lora1.weight
            k_lora = m.k_lora2.weight @ m.k_lora1.weight
            v_lora = m.v_lora2.weight @ m.v_lora1.weight
            out_lora = m.out_lora2.weight @ m.out_lora1.weight
            
            dtype = m.out_proj.weight.dtype

            for new_name, new_param in attention.named_parameters():
                found = False

                for old_name, old_param in m.named_parameters():
                    if new_name == old_name:
                        found = True
                        new_param.data = old_param.data.to(dtype=dtype)
                        break

                if not found:
                    raise ValueError(f"No new parameter found with name {new_name}...")

            del m

            to_replace[n] = attention
            
    # actual replacement
    for n, p in to_replace.items():
        subm = model
        for subn in n.split('.')[:-1]:
            subm = getattr(subm, subn)
        setattr(subm, n.split('.')[-1], p)
        print(f'\treplaced {n}')

    

def absorb_lora(model):
    print('Absorbing lora terms...')

    # create dict with new layers
    to_replace = {}
    for n, m in model.named_modules():
        if isinstance(m, OPTLoraAttention):
            embed_dim = m.embed_dim
            num_heads = m.num_heads
            dropout = m.dropout
            is_decoder = m.is_decoder
            bias = m.k_proj.bias is not None

            attention = OPTAttention(embed_dim, num_heads, dropout, is_decoder, bias)

            q_lora = m.q_lora2.weight @ m.q_lora1.weight
            k_lora = m.k_lora2.weight @ m.k_lora1.weight
            v_lora = m.v_lora2.weight @ m.v_lora1.weight
            out_lora = m.out_lora2.weight @ m.out_lora1.weight
            
            m.q_proj.weight.data += q_lora
            m.k_proj.weight.data += k_lora
            m.v_proj.weight.data += v_lora
            m.out_proj.weight.data += out_lora

            dtype = m.out_proj.weight.dtype

            for new_name, new_param in attention.named_parameters():
                found = False

                for old_name, old_param in m.named_parameters():
                    if new_name == old_name:
                        found = True
                        new_param.data = old_param.data.to(dtype=dtype)
                        break

                if not found:
                    raise ValueError(f"No new parameter found with name {new_name}...")

            del m

            to_replace[n] = attention
            
    # actual replacement
    for n, p in to_replace.items():
        subm = model
        for subn in n.split('.')[:-1]:
            subm = getattr(subm, subn)
        setattr(subm, n.split('.')[-1], p)
        print(f'\treplaced {n}')



# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import math

import torch
from torch import nn

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def eval_model(model, model_str, enc, dev="cuda"):
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False

    enc = enc.input_ids

    enc = enc.to(dev)
    nsamples = enc.numel() // model.seqlen

    losses = 0.0
    for i in range(nsamples):
        if (i % 10) == 0:
            print(f"\tPass {i+1} of {nsamples}")
        batch = enc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)

        out = model(batch)

        if "logits" in out.keys():
            logits = out["logits"]
        else:
            raise ValueError(f"Unknown model out keys:", out.keys())

        loss = nn.CrossEntropyLoss()
        L = loss(logits[:, :-1, :].view(-1, logits.size(-1)), batch[:, 1:].view(-1))

        losses += L.item()

    ppl = math.exp(losses / nsamples)

    model.config.use_cache = use_cache

    outdir = {}
    outdir["ppl"] = ppl

    return outdir


@torch.no_grad()
def eval_model_lowmem(model, model_str, enc, dev="cuda"):
    model.eval()

    enc = enc.input_ids
    nsamples = enc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in model_str:
        layers = model.model.decoder.layers

        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    else:
        layers = model.transformer.h

        model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(dev)
        )

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "alibi": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if not "opt" in model_str:
                cache["alibi"] = kwargs["alibi"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in range(nsamples):
        batch = enc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module.cpu()

    if "opt" in model_str:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
        torch.cuda.empty_cache()
    else:
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.cpu()
        )
        torch.cuda.empty_cache()
        alibi = cache["alibi"]

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for layer_i in range(len(layers)):
        layer = layers[layer_i].to(dev)

        if "opt" in model_str:
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        else:
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

        layers[layer_i] = layer.cpu()

        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if "opt" in model_str:
        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    else:
        model.transformer.ln_f = model.transformer.ln_f.to(dev)

    model.lm_head = model.lm_head.to(dev)

    enc = enc.to(dev)
    sum_nll = 0.0
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)

        if "opt" in model_str:
            if model.model.decoder.final_layer_norm is not None:
                hidden_states = model.model.decoder.final_layer_norm(hidden_states)
            if model.model.decoder.project_out is not None:
                hidden_states = model.model.decoder.project_out(hidden_states)
        else:
            hidden_states = model.transformer.ln_f(hidden_states)

        lm_logits = model.lm_head(hidden_states)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = enc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]

        del hidden_states
        torch.cuda.empty_cache()

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        neg_log_likelihood = loss.float() * model.seqlen

        sum_nll += neg_log_likelihood.item()
    ppl = math.exp(sum_nll / (nsamples * model.seqlen))

    model.config.use_cache = use_cache

    outdir = {}
    outdir["ppl"] = ppl

    return outdir

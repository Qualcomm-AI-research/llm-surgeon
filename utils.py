# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

from pathlib import Path

import psutil
import torch
from pynvml import *
from transformers import LlamaForCausalLM, OPTForCausalLM

ASSERT = False


def full_to_sub(M, dead_rows, dead_cols):
    M_sub = M[~dead_rows, :][:, ~dead_cols]

    if ASSERT:
        M_full = sub_to_full(M_sub, dead_rows, dead_cols)
        assert torch.isclose(M, M_full).all(), f"full_to_sub -> sub_to_full unit test failed..."

    return M_sub


def aug(M, perc_damp=None, damp=None, inplace=False):
    if not inplace:
        M = M.clone()

    dev = M.device

    if damp is None:
        damp = perc_damp * torch.mean(torch.diag(M))  # diagonal damping

    diag = torch.arange(len(M), device=dev)
    M[diag, diag] += damp

    return M


def sub_to_full(M_sub, dead_rows=None, dead_cols=None, unit_test=True):
    assert (dead_rows is not None) or (
        dead_cols is not None
    ), f"Should receive dead_rows, dead_cols or both"
    dev = M_sub.device

    row_indices = torch.where(~dead_rows)[0]
    col_indices = torch.where(~dead_cols)[0]

    M_full_shape = (len(dead_rows), len(dead_cols))

    M_temp = torch.zeros((M_full_shape[0], len(col_indices)), device=dev)
    M_temp[row_indices, :] = M_sub

    M_full = torch.zeros(M_full_shape, device=dev)
    M_full[:, col_indices] = M_temp

    if ASSERT:
        assert not (M_full[dead_rows, :] != 0.0).any(), f"dead rows unit test failed."
        assert not (M_full[:, dead_cols] != 0.0).any(), f"dead cols unit test failed."

    return M_full


def inv(M, perc_damp, sub=True):
    dead = M.diag() == 0.0

    if dead.any() and sub:
        print("[WARNING] DIAGONAL VALUE 0 FOUND IN INV")

        M_out = full_to_sub(M_out, dead, dead)
        aug(M_out, perc_damp, inplace=True)

        M_out = torch.linalg.cholesky(M_out.double())
        M_out = torch.cholesky_inverse(M_out)
    else:
        M_out = aug(M, perc_damp, inplace=False)
        try:
            M_out = torch.linalg.cholesky(M_out.double())
        except:
            M_out = torch.linalg.cholesky(M_out.double())
        M_out = torch.cholesky_inverse(M_out)

    return M_out.float()


def log_usage(writer, base_i, i):
    torch.cuda.empty_cache()
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    gpumem = info.used / (1024**3)

    writer.add_scalar(f"resources/gpumem_{base_i}", gpumem, i)

    cpumem = dict(psutil.virtual_memory()._asdict())["used"] / (1024**3)

    writer.add_scalar(f"resources/cpumem_{base_i}", cpumem, i)

    t = i * 10 + base_i
    writer.add_scalar(f"resources/time", t, t)


def log_model(writer, model, i):
    model_params = torch.cat([x.view(-1).cpu().detach() for x in model.parameters()])

    if isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    else:
        raise NotImplementedError(f"Unknown model type: ", type(model))

    decoder_params = torch.cat([x.view(-1).cpu().detach() for x in layers.parameters()])

    model_sparsity = torch.mean(1.0 * (model_params != 0.0)).item()
    model_zeros = torch.sum(1.0 * (model_params == 0.0)).item()
    model_params = model_params.numel()

    decoder_sparsity = torch.mean(1.0 * (decoder_params != 0.0)).item()
    decoder_zeros = torch.sum(1.0 * (decoder_params == 0.0)).item()
    decoder_params = decoder_params.numel()

    writer.add_scalar(f"stats/sparsity", model_sparsity, i)
    writer.add_scalar(f"stats/zeros", model_zeros, i)
    writer.add_scalar(f"stats/params", model_params, i)

    writer.add_scalar(f"stats/subsparsity", decoder_sparsity, i)
    writer.add_scalar(f"stats/subzeros", decoder_zeros, i)
    writer.add_scalar(f"stats/subparams", decoder_params, i)


def save_curvature_to_dir(
    curvature, model, shot_i, name, save_curvature, path="./curvatures_mini/"
):
    Path(path).mkdir(parents=True, exist_ok=True)
    save_dir = Path(f"{path}/curvature_{shot_i}_{name}")

    print(f"Saving result... [{save_dir}]")
    curvature2 = {}
    for mod, dic in curvature.curvature.items():
        keyname = "ERROR"
        for n, p in model.named_parameters():
            if p is mod.weight:
                keyname = n
                break

        curvature2[keyname] = {}
        for keyval, valval in dic.items():
            curvature2[keyname][keyval] = valval.detach().cpu().clone()

    torch.save(curvature2, save_dir)

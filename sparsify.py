# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch

from modelutils import *
from pruner import Pruner
from threshold import (
    semistructured_threshold,
    structured_threshold,
    unstructured_threshold,
)
from utils import inv

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from transformers import LlamaForCausalLM, OPTForCausalLM


def model_zeros(model):
    params = torch.cat([x.view(-1).cpu().detach() for x in model.parameters()])
    return torch.sum(1.0 * (params == 0.0)).item()


@torch.no_grad()
def sparsify_model(
    model,
    sparsity,
    structures,
    curvature,
    obd,
    update_scale,
    max_correlate,
    damp_g,
    damp_a,
    use_diagonal,
    rank1cost,
    eigenfix,
    addupdate,
    zerofix,
    strictmax,
):
    assert len(structures) > 0, f"Need at least 1 structure to prune..."
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    else:
        raise NotImplementedError(f"Unknown model type: ", type(model))

    print(f"Prune {len(layers)} layers.")

    all_mods = []
    structured_losses = {}
    pruners = {}

    mod_i = 0
    for layer_i, layer in enumerate(layers):
        print(f"Computing losses for layer {layer_i + 1} / {len(layers)}.")
        mods = find_layers(layer)

        for _, mod in mods.items():
            pruner = Pruner(mod, curvature[mod], eigenfix)

            W, G, A, diagonal, dead_rows, dead_cols = pruner.WGA()
            dead = (dead_rows, dead_cols)

            full_W_shape = (
                (mod.weight.shape[0], mod.weight.shape[1] + 1)
                if mod.bias is not None
                else (mod.weight.shape[0], mod.weight.shape[1])
            )

            curvature_dev = G[0].device
            G = [Gi.to(W.device) for Gi in G]
            A = [Ai.to(W.device) for Ai in A]

            if obd:  # or (krank == 2):
                Ginv = None
                Ainv = None
            else:
                Ginv = [inv(x, damp_g) for x in G]
                Ainv = [inv(x, damp_a) for x in A]
            if ("row" in structures) and ("column" in structures):
                row_losses = pruner.local_costs(
                    W,
                    G,
                    A,
                    diagonal,
                    "row",
                    curvature,
                    obd,
                    use_diagonal,
                    rank1cost,
                    damp_g,
                    damp_a,
                    Ginv=Ginv,
                    Ainv=Ainv,
                )
                col_losses = pruner.local_costs(
                    W,
                    G,
                    A,
                    diagonal,
                    "column",
                    curvature,
                    obd,
                    use_diagonal,
                    rank1cost,
                    damp_g,
                    damp_a,
                    Ginv=Ginv,
                    Ainv=Ainv,
                )

                losses = (row_losses, col_losses)
            elif "2:4" in structures:
                losses = pruner.local_costs(
                    W,
                    G,
                    A,
                    diagonal,
                    "element",
                    curvature,
                    obd,
                    use_diagonal,
                    rank1cost,
                    damp_g,
                    damp_a,
                    Ginv=Ginv,
                    Ainv=Ainv,
                )

                assert W.shape == losses.shape, f"W.shape != losses.shape"
            elif "element" in structures:
                losses = pruner.local_costs(
                    W,
                    G,
                    A,
                    diagonal,
                    "element",
                    curvature,
                    obd,
                    use_diagonal,
                    rank1cost,
                    damp_g,
                    damp_a,
                    Ginv=Ginv,
                    Ainv=Ainv,
                )

                assert W.shape == losses.shape, f"W.shape != losses.shape"
            else:
                raise NotImplementedError(f"Unsupported structures: [{structures}]")

            if Ginv is not None:
                Ginv = [x.to(curvature_dev) for x in Ginv]
                Ainv = [x.to(curvature_dev) for x in Ainv]

            G = [x.to(curvature_dev) for x in G]
            A = [x.to(curvature_dev) for x in A]
            structured_losses[mod] = {
                "losses": losses,
                "dead": dead,
                "full_W_shape": full_W_shape,
                "has_bias": mod.bias is not None,
                "Ginv": Ginv,
                "Ainv": Ainv,
            }
            pruners[mod] = pruner

            all_mods.append(mod)
            mod_i += 1

    for mod_i, mod in enumerate(all_mods):
        pruner = pruners[mod]

    if ("row" in structures) and ("column" in structures):
        global_threshold = structured_threshold(structured_losses, sparsity)
    elif "element" in structures:
        global_threshold = unstructured_threshold(structured_losses, sparsity)
    elif "2:4" in structures:
        global_threshold = semistructured_threshold(structured_losses, sparsity)
    else:
        raise NotImplementedError(f"Unsupported structures: [{structures}]")

    for mod_i, mod in enumerate(all_mods):
        pruner = pruners[mod]

    print("Global threshold: ", global_threshold)

    for mod_i, mod in enumerate(all_mods):
        print(f"Pruning module {mod_i + 1} of {len(all_mods)}.")
        pruner = pruners[mod]
        Ginv = structured_losses[mod]["Ginv"]
        Ainv = structured_losses[mod]["Ainv"]
        losses = structured_losses[mod]["losses"]
        full_W_shape = structured_losses[mod]["full_W_shape"]

        pruner.weight_update(
            structures,
            curvature,
            obd,
            global_threshold,
            update_scale,
            max_correlate,
            losses,
            Ginv,
            Ainv,
            use_diagonal,
            addupdate,
            damp_g,
            damp_a,
            zerofix,
            strictmax,
        )
        pruner.free()

    model.config.use_cache = use_cache

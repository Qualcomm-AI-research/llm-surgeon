# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import gc
import math

import numpy as np
import torch
from torch import nn
from transformers import LlamaForCausalLM, OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer

EARLY_BUFFER = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def get_mods(layer):
    mods = []
    if isinstance(layer, OPTDecoderLayer):
        mods.append(layer.self_attn.k_proj)
        mods.append(layer.self_attn.q_proj)
        mods.append(layer.self_attn.v_proj)
        mods.append(layer.self_attn.out_proj)
        mods.append(layer.fc1)
        mods.append(layer.fc2)
    elif isinstance(layer, LlamaDecoderLayer):
        mods.append(layer.self_attn.k_proj)
        mods.append(layer.self_attn.q_proj)
        mods.append(layer.self_attn.v_proj)
        mods.append(layer.self_attn.o_proj)
        mods.append(layer.mlp.gate_proj)
        mods.append(layer.mlp.up_proj)
        mods.append(layer.mlp.down_proj)
    else:
        raise NotImplementedError(f"Unknown decoder layer: {type(layer)}")

    return mods


class Curvature:
    def __init__(self, model):
        if isinstance(model, OPTForCausalLM):
            layers = model.model.decoder.layers
        elif isinstance(model, LlamaForCausalLM):
            layers = model.model.layers
        else:
            raise NotImplementedError(f"Unknown model type: ", type(model))

        self.to_hook = []
        for layer in layers:
            self.to_hook.extend(get_mods(layer))

        self.curvature = {}
        for mod in self.to_hook:
            self.curvature[mod] = {}

    def __getitem__(self, module):
        return self.curvature[module]

    def isfinite(self):
        for key, val in self.curvature.items():
            for key2, val2 in val.items():
                if not val2.isfinite().all():
                    return False
        return True

    def free(self):
        for key, val in self.curvature.items():
            for key2, val2 in val.items():
                del val2
            del val
        del self.curvature
        self.curvature = {}


class Identity(Curvature):
    def __init__(self, model):
        super(Identity, self).__init__(model)

    def __getitem__(self, mod):
        num_in_features = (mod.weight.shape[1] + 1) if mod.bias is not None else mod.weight.shape[1]
        num_out_features = mod.weight.shape[0]

        return {"A_mat_0": torch.eye(num_in_features), "G_mat_0": torch.eye(num_out_features)}


class Activations(Curvature):
    def __init__(self, model, train_dataloader, nsamples=128, dev="cuda", curvature_dev=None):
        super(Activations, self).__init__(model)

        model.eval()

        # set up hooks
        curvature = {}

        if isinstance(model, OPTForCausalLM):
            layers = model.model.decoder.layers
        elif isinstance(model, LlamaForCausalLM):
            layers = model.model.layers
        else:
            raise NotImplementedError(f"Unknown model type: ", type(model))

        def hook_fn_forward(module, inps, out):
            if module in self.to_hook:
                assert len(inps) == 1, f"Length of inputs is {len(inps)}, but should be 1"
                for inp in inps:
                    inp = inp.detach().view(-1, inp.shape[-1])
                    seqlen = inp.shape[0]

                    dtype, device = inp.dtype, inp.device

                    if curvature_dev is None:
                        curvature[module]["A_mat_0"] = curvature[module]["A_mat_0"].to(device)
                        curvature[module]["G_mat_0"] = curvature[module]["G_mat_0"].to(device)

                    if module.bias is not None:
                        inp_bias = torch.zeros(
                            (inp.shape[0], inp.shape[1] + 1), dtype=dtype, device=device
                        )
                        inp_bias[:, :-1] = inp
                        inp_bias[:, -1] = 1.0

                        N_sqrt = math.sqrt(seqlen * nsamples)
                        curvature[module]["A_mat_0"].copy_(
                            torch.einsum("bi,bj->ij", inp_bias / N_sqrt, inp_bias / N_sqrt)
                        )
                    else:
                        N_sqrt = math.sqrt(seqlen * nsamples)
                        curvature[module]["A_mat_0"].copy_(
                            torch.einsum("bi,bj->ij", inp / N_sqrt, inp / N_sqrt)
                        )
            del inps
            del out

            return None

        for param in model.parameters():
            param.requires_grad = False

        hooks = []
        for mod in self.to_hook:
            mod_out_features = mod.out_features
            mod_in_features = mod.in_features + (1 if mod.bias is not None else 0)

            curvature[mod] = {
                "A_mat_0": torch.zeros((mod_in_features, mod_in_features), device=curvature_dev),
                "G_mat_0": torch.eye(mod_out_features, device=curvature_dev),
            }

            for name, param in mod.named_parameters():
                param.requires_grad = True

        for mod in self.to_hook:
            h = mod.register_forward_hook(hook_fn_forward)
            hooks.append(h)

        print(
            f"Passing {len(train_dataloader)} batches of data through model to estimate loss landscape (using activations only)."
        )
        with torch.no_grad():
            for i, batch in enumerate(train_dataloader):
                if i % 10 == 0:
                    print(f"Pass {i+1}/{len(train_dataloader)}")

                if len(batch.return_tensors[0]) == 2:
                    batch = torch.tensor(batch.return_tensors[0][0]).to(dev)
                elif len(batch.return_tensors[0]) > 2:
                    batch = torch.tensor([batch.return_tensors[0]]).to(dev)
                else:
                    raise ValueError(f"Something went wrong parsing the input batch shape")

                out = model(batch)

        print("Done estimating loss landscape curvature.")
        self.curvature = curvature

        del train_dataloader


class KFAC(Curvature):
    def __init__(
        self,
        model,
        train_dataloader,
        fisher_samples,
        krank,
        use_iad=False,
        nsamples=128,
        kpca_iter=1,
        max_outgrad=0.0,
        dev="cuda",
        writer=None,
        shot_i=None,
        diagonal=False,
        buffer_dev=None,
        curvature_dev=None,
        save_curvature=0,
        double=False,
        reuse=None,
    ):
        super(KFAC, self).__init__(model)

        self.krank = krank

        model.eval()

        model.float()

        # set up hooks
        curvature = {}
        curv_buffer = {}

        if isinstance(model, OPTForCausalLM):
            layers = model.model.decoder.layers
        elif isinstance(model, LlamaForCausalLM):
            layers = model.model.layers
        else:
            raise NotImplementedError(f"Unknown model type: ", type(model))

        def hook_fn_forward(module, inps, out):
            if module in self.to_hook:
                assert len(inps) == 1, f"Length of inputs is {len(inps)}, but should be 1"
                for inp in inps:
                    if EARLY_BUFFER:
                        inp = inp.detach().to(buffer_dev).view(-1, inp.shape[-1])
                    else:
                        inp = inp.detach().view(-1, inp.shape[-1])

                    if double:
                        inp = inp.double()

                    if module.bias is not None:
                        dtype, device = inp.dtype, inp.device

                        inp_bias = torch.zeros(
                            (inp.shape[0], inp.shape[1] + 1), dtype=dtype, device=device
                        )
                        inp_bias[:, :-1] = inp
                        inp_bias[:, -1] = 1.0
                        if EARLY_BUFFER:
                            curv_buffer[module]["A"] = inp_bias.to(buffer_dev)
                        else:
                            curv_buffer[module]["A"] = inp_bias
                    else:
                        if EARLY_BUFFER:
                            curv_buffer[module]["A"] = inp.to(buffer_dev)
                        else:
                            curv_buffer[module]["A"] = inp
            del inps
            del out

            return None

        def hook_fn_backward_s(module, inp_grad, out_grad, use_iad, krank_i=0, is_first=False):
            if module in self.to_hook:
                assert len(out_grad) == 1, f"Length of out_grad is {len(out_grad)}, but should be 1"
                for outgrad in out_grad:
                    if max_outgrad > 0:
                        outgrad = torch.clamp(outgrad.detach(), -max_outgrad, max_outgrad)
                        if EARLY_BUFFER:
                            G = outgrad.detach().to(buffer_dev).view(-1, outgrad.shape[-1])
                        else:
                            G = outgrad.detach().view(-1, outgrad.shape[-1])
                    else:
                        if EARLY_BUFFER:
                            G = outgrad.detach().to(buffer_dev).view(-1, outgrad.shape[-1])
                        else:
                            G = outgrad.detach().view(-1, outgrad.shape[-1])

                    if double:
                        G = G.double()

                    G[G != G] = 0.0

                    assert G.isfinite().all(), "Gradient is not all finite!"

                    seqlen = G.shape[0]

                    A = curv_buffer[module]["A"]

                    if (save_curvature > 0) and is_first:
                        G_mini = G[:, :save_curvature]
                        A_mini = A[:, :save_curvature]

                        N_root4 = math.sqrt(math.sqrt(seqlen * nsamples))
                        F_mini = torch.sum(
                            torch.einsum(
                                "bi,bj,bk,bl->bikjl",
                                G_mini / N_root4,
                                G_mini / N_root4,
                                A_mini / N_root4,
                                A_mini / N_root4,
                            ),
                            0,
                        ).reshape(save_curvature**2, save_curvature**2)

                        curvature[module]["F_mini"].add_(F_mini.to(curvature_dev).float().cpu())

                    assert G.isfinite().all(), "G is not all finite!"

                    if diagonal and is_first:
                        N_sqrt = math.sqrt(seqlen * nsamples)
                        G_diag = G**2
                        A_diag = A**2
                        D = torch.einsum("bi,bj->ij", G_diag, A_diag)

                        if buffer_dev is None:
                            curv_buffer[module]["diagonal"] = curv_buffer[module]["diagonal"].to(
                                D.device
                            )

                        curv_buffer[module]["diagonal"].add_(D.to(buffer_dev))

                    if use_iad:
                        assert G.isfinite().all(), f"G is not all finite!"

                        N_sqrt = math.sqrt(seqlen * nsamples)
                        G_new = torch.einsum("bu,bv->uv", G / N_sqrt, G / N_sqrt)
                        A_new = torch.einsum("bu,bv->uv", A / N_sqrt, A / N_sqrt)

                        assert G_new.isfinite().all(), f"G is not all finite!, {seqlen}, {nsamples}"

                        if buffer_dev is None:
                            curv_buffer[module]["A_agg"] = curv_buffer[module]["A_agg"].to(
                                A_new.device
                            )

                        curv_buffer[module]["A_agg"].add_(A_new.to(buffer_dev))
                        del A_new

                    else:

                        def mat(m):
                            size = int(math.sqrt(m.numel()))
                            return m.reshape(size, size)

                        A_vec = curv_buffer[module]["A_vec"].to(dev)

                        N_root4 = math.sqrt(math.sqrt(seqlen * nsamples))
                        G_new = torch.einsum(
                            "bi,ij,bj,bu,bv->uv",
                            A / N_root4,
                            mat(A_vec),
                            A / N_root4,
                            G / N_root4,
                            G / N_root4,
                        )

                        assert G_new.isfinite().all(), "G is not all finite!"

                        for r in range(krank_i):
                            s_res = torch.sum(
                                curvature[module][f"A_mat_{r}"].to(dev) * A_vec / nsamples
                            ) * curvature[module][f"G_mat_{r}"].to(dev)

                            G_new.sub_(s_res)

                        del A

                    if buffer_dev is None:
                        curv_buffer[module]["G_agg"] = curv_buffer[module]["G_agg"].to(G_new.device)

                    assert G_new.isfinite().all(), "G is not all finite!"

                    curv_buffer[module]["G_agg"].add_(G_new.to(buffer_dev))
                    assert curv_buffer[module]["G_agg"].isfinite().all(), "G is not all finite!"

                    del curv_buffer[module]["A"]
                    del G_new
                    del G
            del inp_grad
            del out_grad

            return None

        def hook_fn_backward_a(module, inp_grad, out_grad, use_iad, krank_i=0):
            assert not use_iad, f"backward_a should not be called in use_aid mode"

            if module in self.to_hook:
                assert len(out_grad) == 1, f"Length of out_groud is {len(out_grad)} but should be 1"
                for outgrad in out_grad:
                    if max_outgrad > 0:
                        outgrad = torch.clamp(outgrad.detach(), -max_outgrad, max_outgrad)
                        if EARLY_BUFFER:
                            G = outgrad.detach().to(buffer_dev).view(-1, outgrad.shape[-1])
                        else:
                            G = outgrad.detach().view(-1, outgrad.shape[-1])
                    else:
                        if EARLY_BUFFER:
                            G = outgrad.detach().to(buffer_dev).view(-1, outgrad.shape[-1])
                        else:
                            G = outgrad.detach().view(-1, outgrad.shape[-1])

                    if double:
                        G = G.double()

                    G[G != G] = 0.0

                    seqlen = G.shape[0]

                    A = curv_buffer[module]["A"]

                    if not G.isfinite().all():
                        print("Backward_a has non finite G.")

                    def mat(m):
                        size = int(math.sqrt(m.numel()))
                        return m.reshape(size, size)

                    G_vec = curv_buffer[module]["G_vec"].to(dev)

                    N_root4 = math.sqrt(math.sqrt(seqlen * nsamples))
                    A_new = torch.einsum(
                        "bi,ij,bj,bu,bv->uv",
                        G / N_root4,
                        mat(G_vec),
                        G / N_root4,
                        A / N_root4,
                        A / N_root4,
                    )

                    for r in range(krank_i):
                        a_res = torch.sum(
                            curvature[module][f"G_mat_{r}"].to(dev) * G_vec / nsamples
                        ) * curvature[module][f"A_mat_{r}"].to(dev)

                        A_new.sub_(a_res)

                    if buffer_dev is None:
                        curv_buffer[module]["A_agg"] = curv_buffer[module]["A_agg"].to(A_new)

                    curv_buffer[module]["A_agg"].add_(A_new.to(buffer_dev))

                    del curv_buffer[module]["A"]
                    del A_new
                    del A
                    del G
                del inp_grad
                del out_grad

            return None

        for param in model.parameters():
            param.requires_grad = False

        hooks = []
        for mod in self.to_hook:
            mod_in_features = mod.in_features + (1 if mod.bias is not None else 0)

            if reuse is not None:
                g_reuse = reuse.curvature[mod]["G_mat_0"]
                a_reuse = reuse.curvature[mod]["A_mat_0"]
                g_norm = torch.norm(g_reuse)
                a_norm = torch.norm(a_reuse)

                curv_buffer[mod] = {
                    "G_agg": torch.zeros((mod.out_features, mod.out_features), device=buffer_dev),
                    "A_agg": torch.zeros((mod_in_features, mod_in_features), device=buffer_dev),
                    "G_vec": g_reuse / g_norm,
                    "A_vec": a_reuse / a_norm,
                    "eigval": g_norm * a_norm,
                }
            else:
                curv_buffer[mod] = {
                    "G_agg": torch.zeros((mod.out_features, mod.out_features), device=buffer_dev),
                    "A_agg": torch.zeros((mod_in_features, mod_in_features), device=buffer_dev),
                    "G_vec": torch.ones((mod.out_features, mod.out_features), device=buffer_dev),
                    "A_vec": torch.ones((mod_in_features, mod_in_features), device=buffer_dev),
                    "eigval": 1.0,
                }

            curvature[mod] = {}
            for r in range(krank):
                curvature[mod][f"G_mat_{r}"] = torch.zeros(
                    (mod.out_features, mod.out_features), device=curvature_dev
                )
                curvature[mod][f"A_mat_{r}"] = torch.zeros(
                    (mod_in_features, mod_in_features), device=curvature_dev
                )

            if diagonal:
                curv_buffer[mod]["diagonal"] = torch.zeros(
                    (mod.out_features, mod_in_features), device=buffer_dev
                )
                curvature[mod][f"diagonal"] = torch.zeros(
                    (mod.out_features, mod_in_features), device=curvature_dev
                )

            if save_curvature:
                curvature[mod][f"F_mini"] = torch.zeros(
                    (save_curvature**2, save_curvature**2), device="cpu"
                )

            for name, param in mod.named_parameters():
                param.requires_grad = True

        past_key_values = None

        for krank_i in range(krank):
            use_iad = use_iad and krank_i == 0

            if isinstance(model, OPTForCausalLM):
                layers = model.model.decoder.layers
            elif isinstance(model, LlamaForCausalLM):
                layers = model.model.layers
            else:
                raise NotImplementedError(f"Unknown model type: ", type(model))

            for mod in curv_buffer.keys():
                if reuse is not None:
                    g_reuse = reuse.curvature[mod]["G_mat_0"]
                    a_reuse = reuse.curvature[mod]["A_mat_0"]
                    g_norm = torch.norm(g_reuse)
                    a_norm = torch.norm(a_reuse)
                    curv_buffer[mod]["G_vec"].copy_(g_reuse / g_norm)
                    curv_buffer[mod]["A_vec"].copy_(a_reuse / a_norm)
                else:
                    curv_buffer[mod]["G_vec"].fill_(1.0)
                    curv_buffer[mod]["A_vec"].fill_(1.0)

            for kpca_i in range(kpca_iter):
                kfac_str = "[using KFAC for first rank]" if use_iad else ""
                print(
                    f"K-RANK: {krank_i + 1} of {krank}. Step: {kpca_i + 1} of {kpca_iter}. {kfac_str}"
                )

                for hook in hooks:
                    hook.remove()

                is_first = (kpca_i == 0) and (krank_i == 0)
                for mod in self.to_hook:
                    h = mod.register_forward_hook(hook_fn_forward)
                    hooks.append(h)

                    if kpca_i % 2 == 0:
                        h = mod.register_backward_hook(
                            lambda arg1, arg2, arg3: hook_fn_backward_s(
                                arg1, arg2, arg3, use_iad, krank_i=krank_i, is_first=is_first
                            )
                        )
                    else:
                        h = mod.register_backward_hook(
                            lambda arg1, arg2, arg3: hook_fn_backward_a(
                                arg1, arg2, arg3, use_iad, krank_i=krank_i
                            )
                        )

                    hooks.append(h)

                    for name, param in mod.named_parameters():
                        param.requires_grad = True

                print(
                    f"Passing {len(train_dataloader)} batches of data through model to estimate loss landscape."
                )
                losses = 0.0
                for i, batch in enumerate(train_dataloader):
                    if i % 10 == 0:
                        print(f"\tPass {i+1}/{len(train_dataloader)}")

                    if len(batch.return_tensors[0]) == 2:
                        batch = torch.tensor(batch.return_tensors[0][0]).to(dev)
                    elif len(batch.return_tensors[0]) > 2:
                        batch = torch.tensor([batch.return_tensors[0]]).to(dev)
                    else:
                        raise ValueError(f"Something went wrong parsing the input batch shape")

                    out = model(batch)

                    logits = out["logits"]

                    loss = nn.CrossEntropyLoss()

                    if fisher_samples > 0:
                        sample_ys = torch.multinomial(
                            torch.nn.functional.softmax(
                                logits[:, :-1, :].view(-1, logits.size(-1)).cpu().data, dim=1
                            ),
                            fisher_samples,
                        ).to(dev)

                        L = torch.mean(
                            torch.stack(
                                [
                                    loss(logits[:, :-1, :].view(-1, logits.size(-1)), sample_y)
                                    for sample_y in sample_ys.T
                                ],
                                0,
                            ),
                            0,
                        )
                    else:
                        L = loss(logits[:, :-1, :].view(-1, logits.size(-1)), batch[:, 1:].view(-1))

                    L.backward()

                    losses += L.item()

                print("Done passing data.")

                if diagonal and (krank_i == 0) and (kpca_i == 0):
                    for mod in self.to_hook:
                        diag = curv_buffer[mod]["diagonal"]
                        curvature[mod][f"diagonal"].copy_(diag)

                if use_iad:
                    for mod in self.to_hook:
                        G_agg = curv_buffer[mod]["G_agg"]
                        A_agg = curv_buffer[mod]["A_agg"]

                        g_norm = G_agg.norm()
                        a_norm = A_agg.norm()
                        norm_sqrt = math.sqrt(g_norm * a_norm)

                        if (g_norm == 0.0) or (a_norm == 0.0):
                            curvature[mod][f"G_mat_{krank_i}"].zero_()
                            curvature[mod][f"A_mat_{krank_i}"].zero_()
                        else:
                            curvature[mod][f"G_mat_{krank_i}"].copy_(G_agg / g_norm * norm_sqrt)
                            curvature[mod][f"A_mat_{krank_i}"].copy_(A_agg / a_norm * norm_sqrt)

                    print("[Used IAD to obtain lightweight first Kronecker rank estimate (KFAC).]")
                    break

                if kpca_i % 2 == 0:
                    # update G

                    for mod in self.to_hook:
                        G_agg = curv_buffer[mod]["G_agg"]

                        g_norm = torch.norm(G_agg)
                        if g_norm != 0:
                            G_vec = G_agg / g_norm
                        else:
                            G_vec = G_agg

                        curv_buffer[mod]["G_vec"] = G_vec
                        curv_buffer[mod]["G_agg"].zero_()

                        del G_agg
                else:
                    # update A

                    kpca_errors = []

                    for mod in self.to_hook:
                        G_agg = curv_buffer[mod]["G_agg"]
                        A_agg = curv_buffer[mod]["A_agg"]

                        G_vec = curv_buffer[mod]["G_vec"]
                        eigval = curv_buffer[mod]["eigval"]

                        err = torch.norm(G_agg - eigval * G_vec).cpu().item()

                        a_norm = torch.norm(A_agg)

                        if a_norm != 0:
                            A_vec = A_agg / a_norm
                        else:
                            A_vec = A_agg

                        curv_buffer[mod]["A_vec"] = A_vec
                        curv_buffer[mod]["A_agg"].zero_()

                        eigval = a_norm.abs()

                        curv_buffer[mod]["eigval"] = eigval

                        if (kpca_i + 1) == kpca_iter:
                            if (G_vec[0, 0] < 0) and (A_vec[0, 0] < 0):
                                curvature[mod][f"G_mat_{krank_i}"].copy_(-G_vec * math.sqrt(eigval))
                                curvature[mod][f"A_mat_{krank_i}"].copy_(-A_vec * math.sqrt(eigval))
                            else:
                                curvature[mod][f"G_mat_{krank_i}"].copy_(G_vec * math.sqrt(eigval))
                                curvature[mod][f"A_mat_{krank_i}"].copy_(A_vec * math.sqrt(eigval))

                        kpca_errors.append(err)

                        del G_vec
                        del A_vec
                        del G_agg
                        del A_agg

                    print("[aggregated and power iteration]: ")
                    print("mean errors:", np.mean(kpca_errors), np.std(kpca_errors))
                    print("max errors:", np.max(kpca_errors))
                    if writer is not None:
                        writer.add_scalar(f"kpca_hooks/krank={krank_i}", len(self.to_hook), kpca_i)
                        writer.add_scalar(
                            f"kpca_mean/{krank_i}_shot={shot_i}", np.mean(kpca_errors), kpca_i
                        )
                        writer.add_scalar(
                            f"kpca_max/{krank_i}_shot={shot_i}", np.max(kpca_errors), kpca_i
                        )

                    # clean
                    for p in model.parameters():
                        p.grad = None

                    del batch
                    del out
                    del L

                    gc.collect()

        losses = losses / len(train_dataloader)

        for h in hooks:
            h.remove()

        # free reuse
        if reuse is not None:
            reuse.free()
            del reuse
            gc.collect()

        print(f"Backward perplexity: ", math.exp(losses))

        del curv_buffer

        print("Done estimating loss landscape curvature.")
        self.curvature = curvature

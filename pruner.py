# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import math

import torch

from utils import full_to_sub, sub_to_full, aug

from curvature import Identity, Activations, KFAC

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Pruner:
    def __init__(self, layer, curvature, eigenfix=False):
        self.layer = layer
        self.dev = self.layer.weight.device

        self.curvature = curvature
        self.eigenfix = eigenfix

    def WGA(self, min_val=0.0, sub=True):
        # Retrieve G, A
        G, A = [], []

        krank = len(self.curvature) // 2 # bit hacky way of retrieving kfac here
        for i in range(krank):
            A.append(self.curvature[f'A_mat_{i}'])
            G.append(self.curvature[f'G_mat_{i}'])

        has_bias = self.layer.bias is not None

        # Retrieve W
        W_device = self.layer.weight.device
        curvature_device = A[0].device

        if has_bias:
            W = torch.zeros((self.layer.weight.shape[0], self.layer.weight.shape[1] + 1), device=W_device)
            W[:, :-1] = self.layer.weight.data.float()
            W[:, -1] = self.layer.bias.data.float()
        else:
            W = torch.zeros((self.layer.weight.shape[0], self.layer.weight.shape[1]), device=W_device)
            W.copy_(self.layer.weight.data.float())

        # set very small values to zero
        for i in range(krank):
            G[i].mul_((G[i].abs() > min_val).float())
            A[i].mul_((A[i].abs() > min_val).float())
        W.mul_((W.abs() > min_val).float())

        # set dead rows/cols to zero
        dead_cols = (~W.to(curvature_device).any(0)) | (A[0].diag() == 0)
        dead_rows = (~W.to(curvature_device).any(1)) | (G[0].diag() == 0)

        if 'diagonal' in self.curvature.keys():
            diagonal = self.curvature['diagonal']
        else:
            diagonal = None

        W[dead_rows, :] = 0.0 
        W[:, dead_cols] = 0.0
        for i in range(krank):
            G[i][dead_rows, :] = 0.0
            G[i][:, dead_rows] = 0.0
            A[i][dead_cols, :] = 0.0
            A[i][:, dead_cols] = 0.0

        # copy in
        if has_bias:
            self.layer.weight.data[:, dead_cols[:-1]] = 0.0
            self.layer.bias.data[:, dead_cols[-1]] = 0.0
            self.layer.weight.data[dead_rows, :] = 0.0
        else:
            self.layer.weight.data[:, dead_cols] = 0.0
            self.layer.weight.data[dead_rows, :] = 0.0

        # redo dead_cols, dead_rows (might be slightly more very incidentally)
        if has_bias:
            W_new = torch.zeros((self.layer.weight.shape[0], self.layer.weight.shape[1] + 1), device=W_device)
            W_new[:, :-1] = self.layer.weight.data.float()
            W_new[:, -1] = self.layer.bias.data.float()
        else:
            W_new = torch.zeros((self.layer.weight.shape[0], self.layer.weight.shape[1]), device=W_device)
            W_new.copy_(self.layer.weight.data.float())
        dead_cols = (~W.to(curvature_device).any(0)) | (A[0].diag() == 0)
        dead_rows = (~W.to(curvature_device).any(1)) | (G[0].diag() == 0)

        if sub:
            W = full_to_sub(W, dead_rows, dead_cols)
            G = [full_to_sub(Gi, dead_rows, dead_rows) for Gi in G]
            A = [full_to_sub(Ai, dead_cols, dead_cols) for Ai in A] 
            if diagonal is not None:
                diagonal = full_to_sub(diagonal, dead_rows, dead_cols)

        return W, G, A, diagonal, dead_rows, dead_cols


    def local_costs(self, W, G, A, diagonal, structure, curvature, obd, use_diagonal, rank1cost, damp_g, damp_a, solver=True, Ginv=None, Ainv=None):
        if isinstance(curvature, Identity):
            assert obd, f"OBS not available for magnitude pruning."

        device = W.device

        if device is not None:
            W = W.to(device)

            if rank1cost:
                G[0] = G[0].to(device)
                A[0] = A[0].to(device)
            else:
                G = [x.to(device) for x in G]
                A = [x.to(device) for x in A]

            if Ginv is not None:
                if rank1cost:
                    Ginv[0] = Ginv[0].to(device)
                    Ainv[0] = Ainv[0].to(device)
                else:
                    Ginv = [x.to(device) for x in Ginv]
                    Ainv = [x.to(device) for x in Ainv]
            if diagonal is not None:
                diagonal = diagonal.to(device)
            
        if isinstance(curvature, KFAC) and (curvature.krank == 2) and (not rank1cost):
            damper_g = math.sqrt(math.sqrt(sum([torch.outer(Gi.diag(), Ai.diag()) for Gi, Ai in zip(G, A)]).mean() * damp_g))
            damper_a = math.sqrt(math.sqrt(sum([torch.outer(Gi.diag(), Ai.diag()) for Gi, Ai in zip(G, A)]).mean() * damp_a))

            if diagonal is not None:
                damper_g = math.sqrt(math.sqrt(diagonal.mean() * damp_g))
                damper_a = math.sqrt(math.sqrt(diagonal.mean() * damp_a))

            G0 = aug(G[0], damp=damper_g, inplace=False)
            A0 = aug(A[0], damp=damper_a, inplace=False)
            G1 = aug(G[1], damp=damper_g, inplace=False)
            A1 = aug(A[1], damp=damper_a, inplace=False)

            if solver:
                L_G0 = torch.linalg.cholesky(G0)
                L_A0 = torch.linalg.cholesky(A0)

                G_both = torch.linalg.solve(L_G0, torch.cholesky(G1))
                G_both = G_both @ G_both.T
                A_both = torch.linalg.solve(L_A0, torch.cholesky(A1))
                A_both = A_both @ A_both.T
            else:
                L_G0_inv = torch.linalg.cholesky(G0).inverse()
                L_A0_inv = torch.linalg.cholesky(A0).inverse()

                G_both = L_G0_inv @ G1 @ L_G0_inv
                A_both = L_A0_inv @ A1 @ L_A0_inv
            s1, E1 = torch.linalg.eigh(G_both)
            s2, E2 = torch.linalg.eigh(A_both)

            if solver:
                K1 = torch.linalg.solve(L_G0, E1)
                K2 = torch.linalg.solve(L_A0, E2)
            else:
                K1 = L_G0_inv @ E1
                K2 = L_A0_inv @ E2

            s = torch.outer(s1, s2) + 1

            if self.eigenfix:
                if s.min().item() < 0.0:
                    s = s + s.min().abs().item()

            sinv = 1 / s

        if structure == 'element':
            if isinstance(curvature, Identity):
                losses = 0.5 * (W ** 2)
            elif isinstance(curvature, Activations):
                if obd:
                    losses = 0.5 * (W ** 2) * sum([Ai.diag().reshape(1, -1) for Ai in A])
                else:
                    losses = 0.5 * (W ** 2) / torch.diag(Ainv[0]).reshape(1, -1)
            elif isinstance(curvature, KFAC):
                if obd:
                    if use_diagonal:
                        F_diag = diagonal
                    else:
                        F_diag = sum([torch.outer(Gi.diag(), Ai.diag()) for Gi, Ai in zip(G, A)])
                    losses = 0.5 * (W ** 2) * F_diag
                else:
                    if (curvature.krank == 1) or rank1cost:
                        Finv_diag = torch.outer(Ginv[0].diag(), Ainv[0].diag())
                    elif curvature.krank == 2:
                        Finv_diag = (K1 ** 2) @ sinv @ (K2.T ** 2)

                    losses = 0.5 * (W ** 2) / Finv_diag
            else:
                raise NotImplementedError(f"Unknown curvature:', {curvature}")

        elif structure == 'row':
            if isinstance(curvature, Identity):
                losses = torch.norm(W, p=2, dim=1)
            elif isinstance(curvature, Activations):
                if obd:
                    losses = 0.5 * (W ** 2) * torch.diag(sum(A)).reshape(1, -1)
                else:
                    losses = 0.5 * (W ** 2) / torch.diag(Ainv[0]).reshape(1, -1)
                losses = losses.sum(1)
            elif isinstance(curvature, KFAC):
                if obd:
                    losses = 0.5 * sum([torch.sum((W @ Ai) * W, 1) * Gi.diag() for Gi, Ai in zip(G, A)])
                else:
                    if (curvature.krank == 1) or rank1cost:
                        G0inv_diag = torch.diag(Ginv[0])
                        losses = 0.5 * torch.sum((W @ A[0]) * W, 1) / G0inv_diag
                    elif curvature.krank == 2:
                        R, C = W.shape
    
                        rstep = 5
                        losses = torch.zeros(R, device=K1.device)
                        for ri in range(0, R, rstep):
                            ri_end = min(R, ri + rstep)

                            Wr = W[ri:ri_end]
                            Kr = torch.kron(K1[ri:ri_end], K2).view(-1, C, R*C) # R'C x RC
                            Zr = torch.einsum('riz,rjz->rij', Kr * sinv.view(1, 1, -1), Kr) # R' x C x C

                            if solver:
                                losses[ri:ri_end] = torch.einsum('ri,ri->r', Wr, torch.linalg.solve(Zr.view(-1, C, C), Wr))
                            else:
                                Zr = torch.inverse(Zr) # R' x C x C
                                losses[ri:ri_end] = torch.einsum('ri,rij,rj->r', Wr, Zr.view(-1, C, C), Wr)

                            del Kr
                            del Zr
                            del Wr
                    else:
                        raise NotImplementedError(f"OBS loss for krank > 2 not done yet")
            else:
                raise NotImplementedError(f"Unknown curvature:', {curvature}")
        elif structure == 'column':
            if isinstance(curvature, Identity):
                losses = torch.norm(W, p=2, dim=0)
            elif isinstance(curvature, Activations):
                if obd:
                    losses = 0.5 * (W ** 2) * sum([Ai.diag().reshape(1, -1) for Ai in A])
                else:
                    losses = 0.5 * (W ** 2) / torch.diag(Ainv[0]).reshape(1, -1)
                losses = losses.sum(0)
            elif isinstance(curvature, KFAC):
                if obd:
                    losses = 0.5 * sum([torch.sum((Gi @ W) * W, 0) * Ai.diag() for Gi, Ai in zip(G, A)])
                else:
                    if (curvature.krank == 1) or rank1cost:
                        A0inv_diag = torch.diag(Ainv[0])
                        losses = 0.5 * torch.sum((G[0] @ W) * W, 0) / A0inv_diag
                    elif curvature.krank == 2:
                        R, C = W.shape

                        cstep = 5
                        losses = torch.zeros(C, device=K1.device)
                        for ci in range(0, C, cstep):
                            ci_end = min(C, ci + cstep)

                            Kc = torch.kron(K1, K2[ci:ci_end]).view(R, -1, R*C) # R x C' x RC
                            Wc = W[:, ci:ci_end]

                            Zc = torch.einsum('icz,jcz->cij', Kc * sinv.view(1, 1, -1), Kc) # C' x R x R
    
                            if solver:
                                losses[ci:ci_end] = torch.einsum('ic,ci->c', Wc, torch.linalg.inverse(Zc.view(-1, R, R), Wc.T))
                            else:
                                Zc = torch.inverse(Zc) # C' x R x R
                                losses[ci:ci_end] = torch.einsum('ic,cij,jc->c', Wc, Zc.view(-1, R, R), Wc)

                            del Wc
                            del Kc
                            del Zc
                    else:
                        raise NotImplementedError(f"OBS loss for krank > 2 not done yet")

            else:
                raise NotImplementedError(f"Unknown curvature:', {curvature}")
        else:
            raise NotImplementedError(f"Unknown structure: ", structure)

        losses[losses != losses] = 0.0

        return losses


    def weight_update(self, structures, curvature, obd, global_threshold, update_scale, max_correlate, losses, Ginv, Ainv, use_diagonal, addupdate, damp_g, damp_a, zerofix, strictmax, solver=True, device=None):
        W, G, A, diagonal, dead_rows, dead_cols = self.WGA()

        krank = len(A)

        device = W.device

        if device is not None:
            W = W.to(device)
            G = [x.to(device) for x in G]
            A = [x.to(device) for x in A]
            if Ginv is not None:
                Ginv = [x.to(device) for x in Ginv]
                Ainv = [x.to(device) for x in Ainv]
            if diagonal is not None:
                diagonal = diagonal.to(device)
    
        if diagonal is not None:
            F_diag = sum([Gi.diag() for Gi in G]).view(-1, 1) * sum([Ai.diag() for Ai in A]).view(1, -1)
            diag_res = F_diag - diagonal

        if (('column' in structures) and ('row' in structures)):
            row_losses, col_losses = losses

            assert W.shape[0] == row_losses.shape[0], f"W.shape[0] ({W.shape}) does not match shape of row losses ({row_losses.shape})"
            assert W.shape[1] == col_losses.shape[0], f"W.shape[1] ({W.shape}) does not match shape of column losses ({col_losses.shape})"

            row_mask = row_losses <= global_threshold
            col_mask = col_losses <= global_threshold

            mask = ~torch.outer(~row_mask, ~col_mask) # 'logical OR' outer product

            do_update = (not obd) and (mask.any())

            if do_update:
                pruning_dev = W.device
                print(f'Structured update (max_correlate={max_correlate})')

                r_indices = torch.where(row_mask)[0].cpu()
                c_indices = torch.where(col_mask)[0].cpu()

                if max_correlate == 0:
                    r_idx_split = [r_indices]
                    c_idx_split = [c_indices]
                else:
                    r_idx_split = torch.split(r_indices, max_correlate)
                    c_idx_split = torch.split(c_indices, max_correlate)

                if (len(r_idx_split) == 1) and (len(r_idx_split[0]) == 0):
                    r_idx_split = []
                if (len(c_idx_split) == 1) and (len(c_idx_split[0]) == 0):
                    c_idx_split = []

                if (len(r_idx_split) > 0) or (len(c_idx_split) > 0):
                    R, C = W.shape

                    if krank == 1:
                        s1, K1 = torch.linalg.eigh(G[0] + torch.eye(len(G[0]), device=G[0].device, dtype=G[0].dtype) * damp_g * torch.mean(G[0].diag()))
                        s2, K2 = torch.linalg.eigh(A[0] + torch.eye(len(A[0]), device=A[0].device, dtype=A[0].dtype) * damp_a * torch.mean(A[0].diag()))

                        s = torch.outer(s1, s2)
                    elif krank == 2:
                        damper_g = math.sqrt(math.sqrt(sum([torch.outer(Gi.diag(), Ai.diag()) for Gi, Ai in zip(G, A)]).mean() * damp_g))
                        damper_a = math.sqrt(math.sqrt(sum([torch.outer(Gi.diag(), Ai.diag()) for Gi, Ai in zip(G, A)]).mean() * damp_a))

                        if diagonal is not None:
                            damper_g = math.sqrt(math.sqrt(diagonal.mean() * damp_g))
                            damper_a = math.sqrt(math.sqrt(diagonal.mean() * damp_a))
                        G0 = aug(G[0], damp=damper_g, inplace=False)
                        A0 = aug(A[0], damp=damper_a, inplace=False)
                        G1 = aug(G[1], damp=damper_g, inplace=False)
                        A1 = aug(A[1], damp=damper_a, inplace=False)

                        if solver:
                            L_G0 = torch.linalg.cholesky(G0)
                            L_A0 = torch.linalg.cholesky(A0)

                            G_both = torch.linalg.solve(L_G0, torch.cholesky(G1))
                            G_both = G_both @ G_both.T
                            A_both = torch.linalg.solve(L_A0, torch.cholesky(A1))
                            A_both = A_both @ A_both.T
                        else:
                            L_G0_inv = torch.linalg.cholesky(G0).inverse()
                            L_A0_inv = torch.linalg.cholesky(A0).inverse()

                            G_both = L_G0_inv @ G1 @ L_G0_inv
                            A_both = L_A0_inv @ A1 @ L_A0_inv
                        s1, E1 = torch.linalg.eigh(G_both)
                        s2, E2 = torch.linalg.eigh(A_both)

                        if solver:
                            K1 = torch.linalg.solve(L_G0, E1)
                            K2 = torch.linalg.solve(L_A0, E2)
                        else:
                            K1 = L_G0_inv @ E1
                            K2 = L_A0_inv @ E2

                        s = torch.outer(s1, s2) + 1

                        if self.eigenfix:
                            if s.min().item() < 0.0:
                                s = s + s.min().abs().item()

                        sinv = 1 / s
                    else:
                        raise NotImplementedError(f"Krank > 2 is not supported")
                        
                    W_errs = torch.zeros_like(W, device=pruning_dev)

                    K1 = K1.to(pruning_dev)
                    K2 = K2.to(pruning_dev)

                    for r_i, r_idx in enumerate(r_idx_split):
                        print(f"r_i: {r_i} / {len(r_idx_split)} ({len(r_idx)})")

                        if addupdate:
                            EQw = W[r_idx, :].to(pruning_dev) + W_errs[r_idx, :].to(pruning_dev) # R' x C
                        else:
                            EQw = W[r_idx, :].to(pruning_dev) # R' x C

                        K1r = K1[r_idx] # R' x R

                        if krank == 1:
                            if solver:
                                M1 = (K1r / s1.view(1, -1).to(pruning_dev)) @ K1r.T # R' x R'
                                M2 = (K2 * s2.view(1, -1).to(pruning_dev)) @ K2.T # C x C
                                Mat = torch.linalg.solve(M1, EQw) @ M2.T
                            else:
                                M1 = torch.inverse((K1r / s1.view(1, -1).to(pruning_dev)) @ K1r.T) # R' x R'
                                M2 = (K2 * s2.view(1, -1).to(pruning_dev)) @ K2.T # C x C

                                Mat = M1 @ EQw @ M2.T # R' x C

                            del M1
                            del M2
                        elif krank == 2:
                            Kr = torch.kron(K1r, K2).view(-1, R*C) # R'C x RC

                            Mat = torch.zeros((Kr.shape[0], Kr.shape[0]), device=pruning_dev)

                            istep = 200
                            for i in range(len(Mat)):
                                i_end = min(len(Mat), i + istep)
                                Mat[i:i_end] = (Kr[i:i_end] * sinv.view(1, -1).to(pruning_dev)) @ Kr.T # 1 x R'C?
                            if solver:
                                Mat = torch.linalg.solve(Mat, EQw.view(-1, 1)).view(-1, C) # R' x C
                            else:
                                Mat = torch.inverse(Mat) # R'C x R'C

                                Mat = (Mat @ EQw.view(-1, 1)).view(-1, C) # R' x C

                            del Kr
                        else:
                            raise NotImplementedError(f"Krank > 2 is not supported")

                        W_update = -K1 @ ((K1r.T @ Mat @ K2) / s.to(pruning_dev)) @ K2.T
                        W_update[W_update != W_update] = 0.0
                        W_update[r_idx, :] = -W[r_idx, :].to(W_update.device)

                        W_errs.add_(W_update)

                        del EQw
                        del K1r
                        del Mat

                    for c_i, c_idx in enumerate(c_idx_split):
                        print(f"c_i: {c_i} / {len(c_idx_split)} (len(c_idx)={len(c_idx)})")

                        if addupdate:
                            EQw = W[:, c_idx].to(pruning_dev) + W_errs[:, c_idx].to(pruning_dev) # R' x C
                        else:
                            EQw = W[:, c_idx].to(pruning_dev) # R x C'

                        K2c = K2[c_idx] # C' x C

                        if krank == 1:
                            M1 = (K1 * s1.view(1, -1).to(pruning_dev)) @ K1.T # R x R
                            if solver:
                                M2 = (K2c / s2.view(1, -1).to(pruning_dev)) @ K2c.T # C' x C'

                                Mat = M1 @ torch.linalg.solve(M2, EQw.T).T # R x C'check
                            else:
                                M2 = torch.inverse((K2c / s2.view(1, -1).to(pruning_dev)) @ K2c.T) # C' x C'

                                Mat = M1 @ EQw @ M2.T # R x C'

                            W_update = -K1 @ ((K1.T @ Mat @ K2c) / s.to(pruning_dev)) @ K2.T

                            del M1
                            del M2
                        elif krank == 2:
                            Kc = torch.kron(K1, K2c).view(-1, R*C) # RC' x RC
                            
                            Mat = (Kc * sinv.view(1, -1).to(pruning_dev)) @ Kc.T # RC' x RC'

                            if solver:
                                Mat = torch.linalg.solve(Mat, EQw.view(-1, 1)).view(R, -1) # R x C'
                            else:
                                Mat = torch.inverse(Mat) # RC' x RC'

                                Mat = (Mat @ EQw.view(-1, 1)).view(R, -1) # R x C'

                            
                            W_update = -K1 @ ((K1.T @ Mat @ K2c) / s.to(pruning_dev)) @ K2.T

                            del Kc
                        else:
                            raise NotImplementedError(f"krank > 2 not supported.")

                        W_update[W_update != W_update] = 0.0
                        W_update[:, c_idx] = -W[:, c_idx].to(W_update.device)
                        W_errs.add_(W_update)

                        del EQw
                        del K2c
                        del Mat

                    W.add_(W_errs.to(W.device) * update_scale)

            W[mask] = 0.0
        elif (('element' in structures) or ('2:4' in structures)): # or (krank == 2):
            if 'element' in structures:
                assert W.shape == losses.shape, f"Shape of W ({W.shape}) does not match shape of losses ({losses.shape})"
                mask = losses <= global_threshold
            elif '2:4' in structures:
                assert W.shape == losses.shape, f"Shape of W ({W.shape}) does not match shape of losses ({losses.shape})"

                m, n = 2, 4
                losses_no_bias = losses[:, :-1].reshape(-1, n)

                block_sorted = torch.sort(losses_no_bias, dim=1)[0]
                local_threshold = block_sorted[:, (m - 1)].view(-1, 1)
                block_costs = torch.sum(block_sorted[:, :m], 1) # assumes independence (block cost = sum of element costs)

                cond_local = (losses_no_bias <= local_threshold)
                cond_global = (block_costs <= global_threshold)

                cond = cond_local & cond_global.view(-1, 1)

                mask = torch.zeros_like(losses, dtype=torch.bool)
                mask[:, :-1] = cond.view(losses.shape[0], -1)

            elif (('row' in structures) and ('column' in structures)): 
                # this is not used anymore, because structured pruning uses more efficient dedicated code.
                row_losses, col_losses = losses
                assert (W.shape[0] == len(row_losses)) and (W.shape[1] == len(col_losses)), f"Shape of W ({W.shape}) does not match shape of losses ({row_losses.shape}, {col_losses})"

                row_mask = row_losses <= global_threshold
                col_mask = col_losses <= global_threshold
                
                mask = ~torch.outer(~row_mask, ~col_mask)
            else:
                raise NotImplementedError(f"Unknown structures: {structures}")

            if zerofix:
                to_update = (W != 0.0)

                prune_mask = mask & (~to_update)

                use_svd = True
            else:
                prune_mask = mask

                use_svd = False

            do_update = (not obd) and prune_mask.any()

            if do_update:
                dev = W.device
                pruning_dev = W.device

                EQw = W[prune_mask.cpu()].view(-1, 1).to(pruning_dev) # R x W

                Qi = prune_mask.nonzero()

                indices = torch.arange(0, len(Qi))

                print(f'Unstructured update (max_correlate={max_correlate})')

                if max_correlate == 0:
                    idx_split = [indices]
                else:
                    # split elements into chunks for each row
                    split_sizes_old = torch.unique(Qi[:, 0], return_counts=True)[1]

                    # group in sets that do not exceed max number of correlation
                    split_sizes = []
                    count = 0
                    for i in range(len(split_sizes_old)):
                        num = split_sizes_old[i]
                        
                        if (count + num) > abs(max_correlate):
                            if strictmax and (count > max_correlate):
                                sub_split_sizes = [max_correlate for _ in range(count // max_correlate)] + [count % max_correlate]
                                assert sum(sub_split_sizes) == count, f"{sum(sub_split_sizes)} not equal to {count}"
                                
                                split_sizes.extend(sub_split_sizes)
                            else:
                                split_sizes.append(count)

                            count = num
                        else:
                            count += num
                    split_sizes.append(count)

                    idx_split = torch.split(indices, split_sizes, dim=0)

                if (len(idx_split) == 1) and (len(idx_split[0]) == 0):
                    idx_split = []

                W_errs = torch.zeros_like(W, device=pruning_dev)

                if use_svd:
                    if (krank == 1):
                        
                        s1, K1 = torch.linalg.eigh(G[0] + torch.eye(len(G[0]), device=G[0].device, dtype=G[0].dtype) * damp_g * torch.mean(G[0].diag()))
                        s2, K2 = torch.linalg.eigh(A[0] + torch.eye(len(A[0]), device=A[0].device, dtype=A[0].dtype) * damp_a * torch.mean(A[0].diag()))

                        s = torch.outer(s1, s2)
                    elif krank == 2:
                        damper_g = math.sqrt(math.sqrt(sum([torch.outer(Gi.diag(), Ai.diag()) for Gi, Ai in zip(G, A)]).mean() * damp_g))
                        damper_a = math.sqrt(math.sqrt(sum([torch.outer(Gi.diag(), Ai.diag()) for Gi, Ai in zip(G, A)]).mean() * damp_a))

                        if diagonal is not None:
                            damper_g = math.sqrt(math.sqrt(diagonal.mean() * damp_g))
                            damper_a = math.sqrt(math.sqrt(diagonal.mean() * damp_a))
                        G0 = aug(G[0], damp=damper_g, inplace=False)
                        A0 = aug(A[0], damp=damper_a, inplace=False)
                        G1 = aug(G[1], damp=damper_g, inplace=False)
                        A1 = aug(A[1], damp=damper_a, inplace=False)

                        if solver:
                            L_G0 = torch.linalg.cholesky(G0)
                            L_A0 = torch.linalg.cholesky(A0)

                            G_both = torch.linalg.solve(L_G0, torch.cholesky(G1))
                            G_both = G_both @ G_both.T
                            A_both = torch.linalg.solve(L_A0, torch.cholesky(A1))
                            A_both = A_both @ A_both.T
                        else:
                            L_G0_inv = torch.linalg.cholesky(G0).inverse()
                            L_A0_inv = torch.linalg.cholesky(A0).inverse()

                            G_both = L_G0_inv @ G1 @ L_G0_inv
                            A_both = L_A0_inv @ A1 @ L_A0_inv
                        s1, E1 = torch.linalg.eigh(G_both)
                        s2, E2 = torch.linalg.eigh(A_both)

                        if solver:
                            K1 = torch.linalg.solve(L_G0, E1)
                            K2 = torch.linalg.solve(L_A0, E2)
                        else:
                            K1 = L_G0_inv @ E1
                            K2 = L_A0_inv @ E2

                        s = torch.outer(s1, s2) + 1

                        if self.eigenfix:
                            if s.min().item() < 0.0:
                                s = s + s.min().abs().item()

                        K1 = K1.to(pruning_dev)
                        K2 = K2.to(pruning_dev)
                        s = s.to(pruning_dev)
                    else:
                        raise NotImplementedError(f"Krank > 2 is not supported")

                    if krank == 1:
                        s1 = s1.to(pruning_dev)
                        s2 = s2.to(pruning_dev)

                nonzero_indices = torch.where(W.view(-1) != 0.0)[0].cpu()

                for idx_i, idx in enumerate(idx_split):

                    if len(idx) == 0:
                        print(f'[WARNING] Found empty idx split.')
                        continue

                    # get row/col indices associated with pruned weights
                    Qi_part = Qi[idx]
                    Qir, Qic = Qi_part[:, 0], Qi_part[:, 1]

                    # whether to incorporate updates of previous independent updates in same shot  (recommended)
                    if addupdate:
                        EQw_part = (EQw[idx].to(pruning_dev) + W_errs[Qir, Qic].view(-1, 1).to(pruning_dev))
                    else:
                        EQw_part = EQw[idx].to(pruning_dev)

                    Q_len = len(Qir)

                    if krank == 1:
                        if not use_svd: # direct implementation:
                            Ginv_r = Ginv[0][Qir, :][:, Qir]
                            Ainv_c = Ainv[0][Qic, :][:, Qic]

                            C = Ginv_r * Ainv_c

                            Mat = torch.zeros_like(W, device=pruning_dev) # R x W
                            if len(C) == 1:
                                Mat[Qir, Qic] = ((1 / C) @ EQw_part).view(-1) # Q x 1
                            else:
                                if solver:
                                    Mat[Qir, Qic] = (torch.linalg.solve(C, EQw_part)).view(-1) # Q x 1
                                else:
                                    Mat[Qir, Qic] = (torch.inverse(C) @ EQw_part).view(-1) # Q x 1

                            W_update = -Ginv[0] @ Mat @ Ainv[0]

                            W_errs.add_(W_update)

                        else: # svd implementation:
                            if zerofix:
                                len_Q = len(Qir)

                                Ni_part = to_update.nonzero()

                                print('Ni_part:', Ni_part.shape)
                                if len(Ni_part) == 0:
                                    print('[WARNING] skipping update because Ni_part has zero elements')

                                C = torch.zeros((len_Q, len_Q), device=K1.device)

                                istep = 100
                                for i in range(0, len(Ni_part), istep):
                                    i_end = min(i + istep, len(Ni_part))
                                    print(i, i_end)

                                    Nir, Nic = Ni_part[i:i_end, 0], Ni_part[i:i_end, 1]
                                    print('Nir/Nic:', Nir.shape, Nic.shape)

                                    K1r = K1[Qir, :][:, Nir]
                                    print('K1r:', K1r.shape)
                                    K2c = K2[Qic, :][:, Nic]
                                    print('K2c:', K2c.shape)

                                    C[i:i+i_end] = (K1r * s1[Nir].view(1, -1)) @ K1r.T
                                    C[i:i+i_end] *= (K2c * s2[Nic].view(1, -1)) @ K2c.T
                            else:
                                C = K1r / torch.sqrt(s1.view(1, -1).abs())
                                C = C @ C.T
                                C_tmp = K2c / torch.sqrt(s2.view(1, -1).abs())
                                C.mul_(C_tmp @ C_tmp.T) # Q x Q
                                del C_tmp

                            Mat = torch.zeros_like(W, device=pruning_dev) # R x W
                            if len(C) == 1:
                                Mat[Qir, Qic] = ((1 / C) @ EQw_part).view(-1) # Q x 1
                            else:
                                if solver:
                                    Mat[Qir, Qic] = (torch.linalg.solve(C, EQw_part)).view(-1) # Q x 1
                                else:
                                    Mat[Qir, Qic] = (torch.inverse(C) @ EQw_part).view(-1) # Q x 1

                            W_update = -K1 @ ((K1.T @ Mat @ K2) / s) @ K2.T

                            W_update = W_update * (1.0 * (to_update))
                            to_update[Qir, Qic] = False

                            W_errs.add_(W_update)
                    elif krank == 2:
                        K1r = K1[Qir]
                        K2c = K2[Qic]

                        EK = (K1r.view(Q_len, -1, 1) * K2c.view(Q_len, 1, -1)).view(Q_len, -1) # Q x RC

                        C = EK / torch.sqrt(s.view(1, -1).abs())
                        C = C @ C.T

                        Mat = torch.zeros_like(W, device=pruning_dev) # R x W

                        if len(C) == 1:
                            Mat[Qir, Qic] = ((1 / C) @ EQw_part).view(-1) # Q x 1
                        else:
                            if solver:
                                Mat[Qir, Qic] = (torch.linalg.solve(C, EQw_part)).view(-1) # Q x 1
                            else:
                                Mat[Qir, Qic] = (torch.inverse(C) @ EQw_part).view(-1) # Q x 1

                        W_update = -K1 @ ((K1.T @ Mat @ K2) / s) @ K2.T

                        W_errs.add_(W_update)

                        del EK
                    else:
                        raise NotImplementedError(f"Krank > 2 is not supported")

                    if krank > 1:
                        del K1r
                        del K2c

                    del Mat
                    del W_update

                    del Qi_part
                    del EQw_part


                W.add_(W_errs.to(W.device) * update_scale)

                del EQw
                del Qi

                if krank > 1:
                    del K1
                    del K2

            W[mask] = 0.0
        else:
            raise NotImplementedError(f"Unknown structures: {structures}.")

        W = sub_to_full(W, dead_rows, dead_cols)

        if self.layer.bias is not None:
            self.layer.weight.data.copy_(W[:, :-1])
            self.layer.bias.data.copy_(W[:, -1])
        else:
            self.layer.weight.data.copy_(W)

    def free(self):
        self.curvature = None



# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch

from utils import sub_to_full

""" Threshold functions determine a global threshold scalar based on local losses
    unstructured (for each weight), or structured (for rows and columns) that corresponds
    to the desired target sparsity. """


def unstructured_threshold(all_losses, target_sparsity, device="cpu"):
    print("Computing unstructured threshold...")

    sorted_losses = []
    for _, structured_losses in all_losses.items():
        losses = structured_losses["losses"]
        losses = losses.to(device)

        dead_rows, dead_cols = structured_losses["dead"]

        losses = sub_to_full(losses, dead_rows, dead_cols)

        sorted_losses.append(losses.flatten())

    sorted_losses = torch.cat(sorted_losses)
    sorted_losses, _ = torch.sort(sorted_losses)

    total_params = len(sorted_losses)

    target_i = int(target_sparsity * total_params)

    threshold_value = sorted_losses[target_i].item()

    resulting_sparsity = torch.mean(1.0 * (threshold_value <= sorted_losses)).item()

    print("-------------------")
    print("UNSTRUCTURED THRESHOLDING:")
    print("\tmin/max val", sorted_losses.min().item(), sorted_losses.max().item())
    print("\ttotal params:", total_params)
    print("\ttarget sparsity:", target_sparsity)
    print("\tthreshold value:", threshold_value)
    print("\tresulting sparsity:", resulting_sparsity)
    print("-------------------")

    return threshold_value


def semistructured_threshold(all_losses, target_sparsity, mn=(2, 4), device="cpu"):
    print("Computing semistructured threshold...")

    m, n = mn

    sorted_losses = []
    for _, structured_losses in all_losses.items():
        losses = structured_losses["losses"]
        full_W_shape = structured_losses["full_W_shape"]
        has_bias = structured_losses["has_bias"]
        dead_rows, dead_cols = structured_losses["dead"]

        losses = sub_to_full(losses, dead_rows, dead_cols)
        losses = losses.to(device)

        losses_no_bias = losses[:, :-1] if has_bias else losses
        assert (
            losses_no_bias.shape[1] % n
        ) == 0, f"Expected even number of columns without bias to be divisible by N ({n}). Got {losses.shape}), full_W_shape={full_W_shape}, has_bias={has_bias}."
        losses_no_bias = losses_no_bias.reshape(-1, n)

        block_sorted = torch.sort(losses_no_bias, dim=1)[0]
        block_costs = torch.sum(
            block_sorted[:, :m], 1
        )  # assumes independence (block cost = sum of element costs)

        sorted_losses.append(block_costs.flatten())

    target_block_sparsity = target_sparsity * (n / m)

    sorted_losses = torch.cat(sorted_losses)
    sorted_losses, _ = torch.sort(sorted_losses)

    total_blocks = len(sorted_losses)

    target_i = min(
        int(target_block_sparsity * total_blocks), len(sorted_losses) - 1
    )  # can this edgecase occur if rounding is unlucky? otherwise, remove min() for cleanness

    threshold_value = sorted_losses[target_i].item()

    resulting_block_sparsity = torch.mean(1.0 * (threshold_value <= sorted_losses)).item()
    resulting_sparsity = resulting_block_sparsity * (m / n)

    print("-------------------")
    print("SEMISTRUCTURED THRESHOLDING:")
    print("\ttarget sparsity:", target_sparsity)
    print("\ttarget block sparsity:", target_block_sparsity)
    print("\tmin/max val", sorted_losses.min().item(), sorted_losses.max().item())
    print(f"\ttotal blocks: {total_blocks} (total params: {total_blocks * n})")
    print("\ttarget_i:", target_i)
    print("\tthreshold value:", threshold_value)
    print("\tresulting block sparsity:", resulting_block_sparsity)
    print("\tresulting sparsity:", resulting_sparsity)
    print("-------------------")

    return threshold_value


def structured_threshold(all_losses, target_sparsity, device="cpu"):
    print("Computing structured threshold...")

    sorted_layer_losses = []
    sorted_layer_values = []
    for _, structured_losses in all_losses.items():
        row_losses, col_losses = structured_losses["losses"]

        if device is not None:
            row_losses, col_losses = row_losses.to(device), col_losses.to(device)
        else:
            device = row_losses.device

        dead_rows, dead_cols = structured_losses["dead"]
        size_of_col, size_of_row = structured_losses["full_W_shape"]

        row_losses = sub_to_full(
            row_losses.view(-1, 1), dead_rows, torch.zeros(1, dtype=torch.bool, device=device)
        ).flatten()
        col_losses = sub_to_full(
            col_losses.view(1, -1), torch.zeros(1, dtype=torch.bool, device=device), dead_cols
        ).flatten()

        losses = torch.cat([row_losses, col_losses])

        RL = torch.zeros(len(row_losses) + len(col_losses), device=device)
        CL = torch.zeros_like(RL)
        RL[: len(row_losses)] = 1.0
        CL[len(row_losses) :] = 1.0

        sorted_losses, indices = torch.sort(losses)
        RL = RL[indices]
        CL = CL[indices]

        RL_cumsum = torch.cumsum(RL, 0)
        CL_cumsum = torch.cumsum(CL, 0)

        sorted_values = RL_cumsum * size_of_row + CL_cumsum * size_of_col - RL_cumsum * CL_cumsum

        sorted_layer_losses.append(sorted_losses)
        sorted_layer_values.append(sorted_values)

    all_sorted_losses = torch.cat(sorted_layer_losses)

    all_sorted_values = []
    count = 0
    for sorted_values in sorted_layer_values:
        len_values = sorted_values.numel()
        auged = torch.zeros_like(all_sorted_losses)
        auged[count : count + len_values] = sorted_values
        all_sorted_values.append(auged)

        count += len_values

    all_sorted_losses, indices = torch.sort(all_sorted_losses)
    all_sorted_values = [x[indices] for x in all_sorted_values]

    final_sorted_values = sum([torch.cummax(x, 0)[0] for x in all_sorted_values])

    total_params = final_sorted_values[-1]

    target_value = int(target_sparsity * total_params)

    target_i = torch.where(final_sorted_values <= target_value)[0].max()

    threshold_value = all_sorted_losses[target_i].item()

    print("-------------------")
    print("STRUCTURED THRESHOLDING:")
    print("\ttotal params:", total_params)
    print("\ttarget sparsity:", target_sparsity)
    print("\ttarget params:", target_value)
    print("\tthreshold value:", threshold_value)
    print("-------------------")

    return threshold_value

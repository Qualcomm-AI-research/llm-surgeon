# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
from pathlib import Path

import gc

import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import OPTForCausalLM
from transformers import LlamaForCausalLM

from transformers import DefaultDataCollator

from datautils import get_wikipedia, get_wikitext2, get_code_search_net, get_c4

from sparsify import sparsify_model
from lora import tune_lora, absorb_lora, undo_lora
from curvature import Identity, Activations, KFAC

from utils import log_usage, log_model, save_curvature_to_dir
import os

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# TODO add token here
TOKEN = None


def get_model(model_str):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    if 'opt' in model_str:
        model = OPTForCausalLM.from_pretrained(model_str, torch_dtype='auto', use_auth_token=TOKEN, device_map='auto')
        model.seqlen = model.config.max_position_embeddings
    elif 'llama' in model_str:
        model = LlamaForCausalLM.from_pretrained(model_str, torch_dtype='auto', use_auth_token=TOKEN, device_map='auto')
        model.seqlen = 4096
    else:
        raise NotImplementedError(f"Unknown model: {model_str}")
    
    return model

def get_datasets(model_str, langs, seed, nsamples, seqlen):
    datasets = {}
    for lang in langs:
        if lang == 'wikitext2':
            datasets[lang] = get_wikitext2(model_str, nsamples=nsamples, seed=seed, seqlen=seqlen)
        elif lang == 'c4':
            datasets[lang] = get_c4(model_str, seed=seed, nsamples=nsamples, seqlen=seqlen)
        elif lang == 'de':
            datasets[lang] = get_wikipedia(model_str, lang, nsamples=nsamples, seed=seed, subsets=(2000, 300), seqlen=seqlen)
        elif lang == 'fr':
            datasets[lang] = get_wikipedia(model_str, lang, nsamples=nsamples, seed=seed, subsets=(2000, 300), seqlen=seqlen)
        elif lang == 'it':
            datasets[lang] = get_wikipedia(model_str, lang, nsamples=nsamples, seed=seed, subsets=(2000, 300), seqlen=seqlen)
        elif lang in ('python', 'java', 'go', 'javascript', 'php', 'ruby'):
            datasets[lang] = get_code_search_net(model_str, lang, nsamples=nsamples, seed=seed, seqlen=seqlen)
        else:
            raise NotImplementedError(f"Unknown language: [{lang}]")
    return datasets


def main(
        model_str, seed, eval_langs, train_lang, lr, epochs, sparsity, structures, curvature, shots, save_model,
        obd, lora, lora_epochs, lora_lr, batch_size, lora_subset, fisher_samples, save_curvature, krank, kpca_iter,
        nsamples, max_outgrad, use_iad, diagonal, update_scale, buffer_dev, curvature_dev, max_correlate, newdata,
        lowmem, double, schedule, damp_g, damp_a, use_diagonal, rank1cost, reuse, eigenfix, log_lora, addupdate,
        skip, zerofix8, strictmax, log_root):
    if lowmem:
        from eval import eval_model_lowmem as eval_model
    else:
        from eval import eval_model

    curvature_str = f'_{curvature}'
    sparsity_str = f'_S={sparsity}' if sparsity != 1.0 else ''
    seed_str = f'seed={seed}' if seed != 100 else ''
    shot_str = '' if shots == 1 else f'_{shots}'
    model_str2 = model_str.replace('/', '-')
    obd_str = '_obd' if obd else '_obs'
    structure_str = '_'.join(structures)
    lora_str = '_lora' if lora else ''
    lora_lr_str = f'_lora_lr={lora_lr}' if lora_lr != 0.0001 else ''
    lsubset_str = f'_lsubset={lora_subset}' if lora_subset != 0.025 else ''
    batch_str = f'_B={batch_size}' if batch_size > 1 else ''
    fsamples_str = f'_F={fisher_samples}'
    kpca_str = f'_krank={krank}_{kpca_iter}'
    nsamples_str = f'_n={nsamples}' if nsamples != 128 else ''
    maxoutgrad_str = f'_maxoutgrad={max_outgrad}' if max_outgrad > 0 else ''
    useiad_str = f'_useiad' if use_iad else '_nearest'
    diag_str = f'_diagonal_use={use_diagonal}' if diagonal else ''
    newdata_str = f'_newdata' if newdata else ''
    us_str = f'_us={update_scale}' if update_scale != 1.0 else ''
    mcor_str = f'_maxcorrelate={max_correlate}' if max_correlate != 0 else ''
    schedule_str = f'_schedule={schedule}'
    double_str = f'_double' if double else ''
    damp_str = f'_dampg={damp_g}_dampa={damp_a}'
    rank1cost_str = f'_rank1cost' if rank1cost else ''
    reuse_str = f'_reuse' if reuse else ''
    eigenfix_str = f'_eigenfix' if eigenfix else ''
    loglora_str = f'_loglora' if log_lora else ''
    addupdate_str = f'_addupdate' if addupdate else ''
    skip_str = f'_skip={skip}' if skip > 0 else ''
    zerofix8_str = f'_zerofix8' if zerofix8 else ''
    strictmax_str = f'_strictmax' if strictmax else ''
    name = f'N_{seed_str}_{model_str2}{sparsity_str}_{train_lang}_{structure_str}{curvature_str}{shot_str}{obd_str}{lora_str}{batch_str}{lsubset_str}{fsamples_str}{kpca_str}{nsamples_str}{maxoutgrad_str}{useiad_str}{diag_str}{us_str}{mcor_str}{newdata_str}{double_str}{schedule_str}{damp_str}{rank1cost_str}{reuse_str}{eigenfix_str}{loglora_str}{addupdate_str}{skip_str}{zerofix8_str}{strictmax_str}'
    writer = SummaryWriter(f"{log_root}/runs/{name}")

    curvature_type = curvature

    model = get_model(model_str)
    model.eval()

    model = model.float()

    print(f'Loaded model: {model_str}')
    print(f'\tSeqlen: {model.seqlen}')

    print("Loading datasets...")
    total_nsamples = nsamples * shots if newdata else nsamples
    datasets = get_datasets(model_str, eval_langs, seed, total_nsamples, model.seqlen)

    for lang, (_, testenc) in datasets.items():
        test_outdir = eval_model(model, model_str, testenc)
        print(f'Test  PPL [{lang}]:', test_outdir['ppl'])
        writer.add_scalar(f'test_ppl/{lang}', test_outdir['ppl'], 0)

    if save_model:
        save_dir = Path(f'{log_root}/checkpoints')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'{name}_base'
        print(f"Saving result... [{save_path}]")
        torch.save(model.state_dict(), save_path)

    trainenc, _ = datasets[train_lang]

    log_model(writer, model, 0)
    log_usage(writer, 1, 0)

    if len(structures) > 0:
        data_collator = DefaultDataCollator

        for shot_i in range(1, shots + 1):
            if shot_i <= skip:
                print(f"Skipping shot {shot_i}")
                continue
                
            if newdata:
                assert len(trainenc) == nsamples * shots, f"Train encodings seem to have wrong size {len(trainenc)}. Expected (nsamples*shots)={nsamples * shots}."
                start_i = shot_i * nsamples
                stop_i = shot_i * nsamples + nsamples
            else:
                assert len(trainenc) == nsamples, f"Train encodings seem to have wrong size {len(trainenc)}. Expected nsamples={nsamples}."
                start_i = 0
                stop_i = nsamples

            if ((shot_i - 1) % 8) == 0:
                if log_lora:
                    print('LORA CHECKER:', shot_i)
                    _, testenc = datasets[train_lang]
                    tune_lora(model, model_str, trainenc[start_i:stop_i], testenc, writer, f"log_shot={shot_i}", 1.0, lr=lora_lr, n_epochs=lora_epochs, batch_size=batch_size)

                    undo_lora(model)

            train_dataloader = DataLoader(trainenc[start_i:stop_i], shuffle=True, batch_size=1, num_workers=4, collate_fn=data_collator)

            torch.cuda.empty_cache()

            print('Sparsifying structures:', structures)

            print('Collecting gradients...')

            log_usage(writer, 2, shot_i)

            if curvature_type == 'identity':
                curvature = Identity(model)
            elif curvature_type == 'activations':
                curvature = Activations(model, train_dataloader, nsamples=nsamples, curvature_dev=curvature_dev)
            elif curvature_type == 'kfac':
                if reuse and (shot_i > (skip + 1)):
                    reuse_kpca_iter = 6
                    curvature = KFAC(model, train_dataloader, fisher_samples, krank, use_iad=use_iad, nsamples=nsamples, kpca_iter=reuse_kpca_iter, max_outgrad=max_outgrad, dev='cuda', writer=writer, shot_i=shot_i, diagonal=diagonal, buffer_dev=buffer_dev, curvature_dev=curvature_dev, save_curvature=save_curvature, double=double, reuse=curvature)
                else:
                    curvature = KFAC(model, train_dataloader, fisher_samples, krank, use_iad=use_iad, nsamples=nsamples, kpca_iter=kpca_iter, max_outgrad=max_outgrad, dev='cuda', writer=writer, shot_i=shot_i, diagonal=diagonal, buffer_dev=buffer_dev, curvature_dev=curvature_dev, save_curvature=save_curvature, double=double)
            else:
                raise NotImplementedError(f"Unknown curvature: {curvature}")

            if diagonal:
                for mod_i, mod in enumerate(curvature.curvature.keys()):
                    diag = curvature.curvature[mod]['diagonal'].view(-1)
                    
                    G, A = [], []
                    for i in range(krank):
                        G.append(curvature.curvature[mod][f'G_mat_{i}'])
                        A.append(curvature.curvature[mod][f'A_mat_{i}'])
                        kron_diag = sum([torch.kron(G[j].diag(), A[j].diag()) for j in range(i+1)]).view(-1)
                        print('diag:', kron_diag.shape, diag.shape)
                        diag_diff = torch.mean((kron_diag - diag) ** 2).item()
                        writer.add_scalar(f'diag/mse_diag_mod={mod_i}_shot={shot_i}', diag_diff, i)

            log_usage(writer, 3, shot_i)

            if save_curvature > 0:
                save_curvature_to_dir(curvature, model, shot_i, name, save_curvature)

            log_usage(writer, 4, shot_i)

            if not curvature.isfinite():
                raise ValueError(f"Found non-finite value in curvature.")

            print('Pruning...')
            if schedule == 'linear':
                shot_sparsity = shot_i * sparsity / shots
            else:
                raise NotImplementedError(f"Unknown schedule for pruning shots: {schedule}. Try using linear instead.")

            sparsify_model(model, shot_sparsity, structures, curvature, obd, update_scale, max_correlate, damp_g, damp_a, use_diagonal, rank1cost, eigenfix, addupdate, zerofix8, strictmax)

            log_usage(writer, 5, shot_i)

            if not reuse:
                # free curvature
                curvature.free()
                del curvature
                gc.collect()

            log_model(writer, model, shot_i)
            log_usage(writer, 6, shot_i)

            print('Evaluating...')
            with torch.no_grad():
                for lang, (_, testenc) in datasets.items():
                    test_ppl = eval_model(model, model_str, testenc)['ppl']
                    writer.add_scalar(f'test_ppl/{lang}', test_ppl, shot_i)
                    print(f'[shot {shot_i}] test PPL:', test_ppl)

            log_usage(writer, 7, shot_i)

            if lora:
                _, testenc = datasets[train_lang]
                print('LoRA finetuning ...')
                tune_lora(model, model_str, trainenc[start_i:stop_i], testenc, writer, f"interleaved_shot={shot_i}", lora_subset, log=False, lr=lora_lr, n_epochs=1, batch_size=batch_size)

                print('Absorbing LoRA...')
                absorb_lora(model)

                with torch.no_grad():
                    for lang, (_, testenc) in datasets.items():
                        test_outdir = eval_model(model, model_str2, testenc)
                        print(f'Shot lora test  PPL [{lang}]:', test_outdir['ppl'])
                        writer.add_scalar(f'shot_lora_test_ppl/{lang}', test_outdir['ppl'])
                        del test_outdir

                    
            if shot_i % 8 == 0:
                if save_model:
                    print(f'Saving checkpoint at shot {shot_i}...')
                    save_dir = Path(f'{log_root}/checkpoints')
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f'{name}_saveshot={shot_i}'
                    print(f"Saving result... [{save_path}]")
                    torch.save(model.state_dict(), save_path)

            log_usage(writer, 8, shot_i)

            del train_dataloader
            gc.collect()

        log_usage(writer, 9, shot_i)


    if lora:
        print(f"Evaluate test set performance before final lora epoch...")

        for lang, (_, testenc) in datasets.items():
            test_outdir = eval_model(model, model_str2, testenc)
            print(f'Pruned test  PPL [{lang}]:', test_outdir['ppl'])
            writer.add_scalar(f'final_prelora_test_ppl/{lang}', test_outdir['ppl'])
            del test_outdir

        _, testenc = datasets[train_lang]
        tune_lora(model, model_str, trainenc[start_i:stop_i], testenc, writer, "final", 1.0, lr=lora_lr, n_epochs=lora_epochs, batch_size=batch_size)

        print('Absorbing LoRA...')
        absorb_lora(model)

        print(f"Evaluate final model on test sets...")
        for lang, (_, testenc) in datasets.items():
            test_outdict = eval_model(model, model_str2, testenc)
            print(f'Final test  PPL [{lang}]:', test_outdict['ppl'])
            writer.add_scalar(f'final_postlora_test_ppl/{lang}', test_outdict['ppl'])
            del test_outdict

    if save_model:
        save_dir = Path(f'{log_root}/checkpoints/')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'{name}_final'
        print(f"Saving result... [{save_path}]")
        torch.save(model.state_dict(), save_path)

    writer.close()


parser = argparse.ArgumentParser(description='llm-compression')
parser.add_argument('--model_str', type=str, default='facebook/opt-125m')
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--eval_langs', type=str, nargs='+', default=['wikitext2'])
parser.add_argument('--train_lang', type=str, default='wikitext2')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--sparsity', type=float, default=0.5)
parser.add_argument('--structures', nargs='+', default=['row', 'column'])
parser.add_argument('--curvature', type=str, default='kfac')
parser.add_argument('--buffer_dev', type=str, default=None)
parser.add_argument('--curvature_dev', type=str, default='cpu')
parser.add_argument('--shots', type=int, default=40)
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--obd', action='store_true')
parser.add_argument('--lora_epochs', type=int, default=5)
parser.add_argument('--lora_lr', type=float, default=0.0001)
parser.add_argument('--lora', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lora_subset', type=float, default=0.025)
parser.add_argument('--fisher_samples', type=int, default=0)
parser.add_argument('--save_curvature', type=int, default=0)
parser.add_argument('--krank', type=int, default=1)
parser.add_argument('--kpca_iter', type=int, default=1)
parser.add_argument('--nsamples', type=int, default=128)
parser.add_argument('--max_outgrad', type=float, default=0.0)
parser.add_argument('--use_iad', action='store_true')
parser.add_argument('--diagonal', action='store_true')
parser.add_argument('--use_diagonal', action='store_true')
parser.add_argument('--newdata', action='store_true')
parser.add_argument('--lowmem', action='store_true')
parser.add_argument('--update_scale', type=float, default=1.0)
parser.add_argument('--max_correlate', type=int, default=0)
parser.add_argument('--double', action='store_true')
parser.add_argument('--schedule', type=str, default='linear')

parser.add_argument('--damp_g', type=float, default=0.01)
parser.add_argument('--damp_a', type=float, default=0.01)
parser.add_argument('--eigenfix', action='store_true')
parser.add_argument('--rank1cost', action='store_true')

parser.add_argument('--reuse', action='store_true')
parser.add_argument('--log_lora', action='store_true')
parser.add_argument('--addupdate', action='store_true')
parser.add_argument('--skip', type=int, default=0)
parser.add_argument('--zerofix8', action='store_true')
parser.add_argument('--strictmax', action='store_true')
parser.add_argument('--log_root', type=str, required=False, default=os.environ.get('LOG_ROOT', None))

args = parser.parse_args()

if args.log_root is None:
    raise ValueError("Either --log_root should be passed or LOG_ROOT should be set in the environment")

print(args)

main(**vars(args))


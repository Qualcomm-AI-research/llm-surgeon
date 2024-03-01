#!/bin/bash

cd ../..

# Llama-v2, 7B
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature activations --model_str $LLAMA_V2_ROOT/7B --shots 24 --structures element --krank 1 --use_iad --obd

# Llama-v2, 13B
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature activations --model_str $LLAMA_V2_ROOT/13B --shots 24 --structures element --krank 1 --use_iad --obd

# OPT, 125m
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature activations --model_str facebook/opt-125m --shots 24 --structures element --krank 1 --use_iad --obd

# OPT, 1.3b
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature activations --model_str facebook/opt-1.3b --shots 24 --structures element --krank 1 --use_iad --obd

# OPT, 2.7b
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature activations --model_str facebook/opt-2.7b --shots 24 --structures element --krank 1 --use_iad --obd

# OPT, 6.7b
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature activations --model_str facebook/opt-6.7b --shots 24 --structures element --krank 1 --use_iad --obd



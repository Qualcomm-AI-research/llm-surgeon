#!/bin/bash

cd ../..

# structured
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str facebook/opt-125m --shots 40 --structures row column --krank 1 --kpca_iter 10 --nsamples 128 --use_iad --obd
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str facebook/opt-1.3b --shots 40 --structures row column --krank 1 --kpca_iter 10 --nsamples 128 --use_iad --obd
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str facebook/opt-2.7b --shots 40 --structures row column --krank 1 --kpca_iter 10 --nsamples 128 --use_iad --obd
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str facebook/opt-6.7b --shots 40 --structures row column --krank 1 --kpca_iter 10 --nsamples 128 --use_iad --obd
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str $LLAMA_V2_ROOT/7B --shots 40 --structures row column --krank 1 --kpca_iter 10 --nsamples 128 --use_iad --obd
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str $LLAMA_V2_ROOT/13B --shots 40 --structures row column --krank 1 --kpca_iter 10 --nsamples 128 --use_iad --obd



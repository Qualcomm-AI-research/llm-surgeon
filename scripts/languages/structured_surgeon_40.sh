#!/bin/bash

cd ../..

# structured

python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 fr de it --sparsity 0.5 --curvature kfac --model_str facebook/opt-125m --shots 40 --structures column row --krank 1 --kpca_iter 10 --nsamples 128 --use_iad --max_correlate 2000 --save_model --damp_g 0.1
python $PROJECT_ROOT/surgeon.py --train_lang fr --eval_langs wikitext2 fr de it --sparsity 0.5 --curvature kfac --model_str facebook/opt-125m --shots 40 --structures column row --krank 1 --kpca_iter 10 --nsamples 128 --use_iad --max_correlate 2000 --save_model --damp_g 0.1
python $PROJECT_ROOT/surgeon.py --train_lang de --eval_langs wikitext2 fr de it --sparsity 0.5 --curvature kfac --model_str facebook/opt-125m --shots 40 --structures column row --krank 1 --kpca_iter 10 --nsamples 128 --use_iad --max_correlate 2000 --save_model --damp_g 0.1
python $PROJECT_ROOT/surgeon.py --train_lang it --eval_langs wikitext2 fr de it --sparsity 0.5 --curvature kfac --model_str facebook/opt-125m --shots 40 --structures column row --krank 1 --kpca_iter 10 --nsamples 128 --use_iad --max_correlate 2000 --save_model --damp_g 0.1



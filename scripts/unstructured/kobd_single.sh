#!/bin/bash

cd ../..

# Llama-v2, 7B
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.1 --curvature kfac --model_str $LLAMA_V2_ROOT/7B --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.2 --curvature kfac --model_str $LLAMA_V2_ROOT/7B --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature kfac --model_str $LLAMA_V2_ROOT/7B --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.4 --curvature kfac --model_str $LLAMA_V2_ROOT/7B --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str $LLAMA_V2_ROOT/7B --shots 1 --structures element --use_iad --obd --damp_g 0.1

# Llama-v2, 13B
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.1 --curvature kfac --model_str $LLAMA_V2_ROOT/13B --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.2 --curvature kfac --model_str $LLAMA_V2_ROOT/13B --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature kfac --model_str $LLAMA_V2_ROOT/13B --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.4 --curvature kfac --model_str $LLAMA_V2_ROOT/13B --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str $LLAMA_V2_ROOT/13B --shots 1 --structures element --use_iad --obd --damp_g 0.1

# OPT, 125m
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.1 --curvature kfac --model_str facebook/opt-125m --shots 1 --structures element --use_iad --obd --damp_g 0.1
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.2 --curvature kfac --model_str facebook/opt-125m --shots 1 --structures element --use_iad --obd --damp_g 0.1
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature kfac --model_str facebook/opt-125m --shots 1 --structures element --use_iad --obd --damp_g 0.1
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.4 --curvature kfac --model_str facebook/opt-125m --shots 1 --structures element --use_iad --obd --damp_g 0.1
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str facebook/opt-125m --shots 1 --structures element --use_iad --obd --damp_g 0.1

# OPT, 1.3b
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.1 --curvature kfac --model_str facebook/opt-1.3b --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.2 --curvature kfac --model_str facebook/opt-1.3b --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature kfac --model_str facebook/opt-1.3b --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.4 --curvature kfac --model_str facebook/opt-1.3b --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str facebook/opt-1.3b --shots 1 --structures element --use_iad --obd --damp_g 0.1

# OPT, 2.7b
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.1 --curvature kfac --model_str facebook/opt-2.7b --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.2 --curvature kfac --model_str facebook/opt-2.7b --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature kfac --model_str facebook/opt-2.7b --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.4 --curvature kfac --model_str facebook/opt-2.7b --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str facebook/opt-2.7b --shots 1 --structures element --use_iad --obd --damp_g 0.1

# OPT, 6.7b
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.1 --curvature kfac --model_str facebook/opt-6.7b --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.2 --curvature kfac --model_str facebook/opt-6.7b --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature kfac --model_str facebook/opt-6.7b --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.4 --curvature kfac --model_str facebook/opt-6.7b --shots 1 --structures element --use_iad --obd --damp_g 0.1
#python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str facebook/opt-6.7b --shots 1 --structures element --use_iad --obd --damp_g 0.1



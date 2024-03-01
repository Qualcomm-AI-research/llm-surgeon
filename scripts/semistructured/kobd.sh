#!/bin/bash

cd ../..

# semi-structured
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str facebook/opt-125m --shots 1 --structures 2:4 --use_iad --obd --damp_g 0.1
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str facebook/opt-1.3b --shots 1 --structures 2:4 --use_iad --obd --damp_g 0.1
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str facebook/opt-2.7b --shots 1 --structures 2:4 --use_iad --obd --damp_g 0.1
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str facebook/opt-6.7b --shots 1 --structures 2:4 --use_iad --obd --damp_g 0.1
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str $LLAMA_V2_ROOT/7B --shots 1 --structures 2:4 --use_iad --obd --damp_g 0.1
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.5 --curvature kfac --model_str $LLAMA_V2_ROOT/13B --shots 1 --structures 2:4 --use_iad --obd --damp_g 0.1



#!/bin/bash

cd ../..

# structured LOBD
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature activations --model_str facebook/opt-125m --shots 24 --structures row column --use_iad --obd
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature activations --model_str facebook/opt-1.3b --shots 24 --structures row column --use_iad --obd
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature activations --model_str facebook/opt-2.7b --shots 24 --structures row column --use_iad --obd
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature activations --model_str facebook/opt-6.7b --shots 24 --structures row column --use_iad --obd
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature activations --model_str $LLAMA_V2_ROOT/7B --shots 24 --structures row column --use_iad --obd
python $PROJECT_ROOT/surgeon.py --train_lang wikitext2 --eval_langs wikitext2 --sparsity 0.3 --curvature activations --model_str $LLAMA_V2_ROOT/13B --shots 24 --structures row column --use_iad --obd



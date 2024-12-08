### Install Custom oe-eval 
```sh
git clone git@github.com:allenai/oe-eval-internal.git olmo-repos/oe-eval-internal
cd olmo-repos/olmo
git checkout paraphrase # get current project branch
pip install -e . # [dev]

# (for vllm support) install nightly vllm
mkdir .vllm-install && cd .vllm-install
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

### Converting OLMo Checkpoints
```sh
# install olmo
git clone git@github.com:allenai/OLMo.git olmo-repos/olmo
cd olmo-repos/olmo
pip install -e .

# install nightly transformers
pip install git+https://github.com/huggingface/transformers

# OLMo 1 models (pre-peteish)
INPUT_DIR=/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81352-unsharded
OUTPUT_DIR=/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81352-unsharded-hf
TOKENIZER_PATH=/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/latest/tokenizer.json
python scripts/convert_olmo_to_hf_new.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --tokenizer_json_path $TOKENIZER_PATH

# OLMo 2 models (post-peteish)
INPUT_DIR=/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81352-unsharded
OUTPUT_DIR=/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81352-unsharded-hf
TOKENIZER_PATH=/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/latest/tokenizer.json
python scripts/convert_olmo2_to_hf.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --tokenizer_json_path $TOKENIZER_PATH

# Convert in a batch
conda activate metaeval
/root/ai2/metaeval/convert_checkpoints_peteish.sh

# Detatch from current session
nohup /root/ai2/metaeval/convert_checkpoints_peteish.sh > out.out 2>&1 &
tail -f out.out
```

### Launching & Processing Evals
```sh
python scripts/launch_evals.py # launch evals on beaker
python download/aws.py # sync from s3
python download/preprocess.py # convert to .parquet

# Detatch from current session
nohup python preprocess.py > /tmp/out.out 2>&1 & tail -f /tmp/out.out
```

### Install Ladder Model Code
```sh
git clone https://github.com/allenai/OLMo/ olmo-repos/olmo
cd olmo-repos/olmo
git checkout ladder-1xC
pip install -e .

# Example: Run variance analysis + predictions
python scripts/scaling/variance_analysis.py -k v2_main_variance -c scripts/scaling/final_variance.json -o figure/peteish-moreeval/variance.pdf --last_n_points 10 --run_prediction

python scripts/scaling/step2.py -k v2_main -c scripts/scaling/step2.json -o figure/peteish-moreeval/step2_main.pdf --skip_perc 0.1 --moving_avg 5
```
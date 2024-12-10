### Install Custom oe-eval 
```sh
git clone git@github.com:allenai/oe-eval-internal.git olmo-repos/oe-eval-internal
cd olmo-repos/oe-eval-internal/
git checkout paraphrase # get current project branch
pip install -e . # [dev] # --no-deps

# (for vllm support) install nightly vllm=
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

### Converting OLMo Checkpoints
```sh
# install olmo
git clone git@github.com:allenai/OLMo.git olmo-repos/OLMo
cd olmo-repos/OLMo
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
nohup /root/ai2/metaeval/convert_checkpoints_peteish.sh > out.out 2>&1 & tail -f out.out
```

### Launching & Processing Evals
```sh
python scripts/launch_evals.py # launch evals on beaker
python analysis/download/aws.py # sync from s3
python analysis/download/preprocess.py # convert to .parquet

# Detatch from current session
nohup python analysis/download/aws.py > /tmp/out.out 2>&1 & tail -f /tmp/out.out
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

```sh
# A failed attempt to increase memory swap size
free -h
sudo fallocate -l 50G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h
```
import argparse
import os
from pathlib import Path
import sys
import torch
import hf_olmo
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from analysis.utils.constants_model_ckpts import MODEL_LIST_FINAL_SIX_CKPTS, DATADECIDE_FINAL_FIVE_CKPTS

FINEWEB_CKPTS = [
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step0-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step2500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step5000-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step7500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step10000-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step12500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step15000-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step17500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step20000-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step22500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step25000-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step27500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step30000-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step32500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step35000-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step37500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step40000-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step42500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step45000-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step47500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step50000-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step52500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step55000-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step57500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step60000-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step62500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step65000-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step67500-unsharded-hf",
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step69369-unsharded-hf",
]


WEKA_PATH = "/oe-eval-default/"

def load_model_paths(file_path):
    """Load model paths from a file."""
    with open(file_path, "r") as f:
        return [line.strip().replace("weka://oe-eval-default/", WEKA_PATH) for line in f if line.strip()]


def weightspace_average_model(model_paths, model_class, device_map="auto"):
    """Perform weightspace averaging of models."""
    averaged_weights = None
    num_models = len(model_paths)

    for i, path in enumerate(model_paths):
        print(f'({i}/{len(model_paths)}) Loading model at: {path}')
        model = model_class.from_pretrained(path, device_map=device_map)
        model_weights = model.state_dict()

        if averaged_weights is None:
            averaged_weights = {key: torch.zeros_like(value) for key, value in model_weights.items()}
        for key in model_weights:
            averaged_weights[key] += model_weights[key]

        del model
        torch.cuda.empty_cache()

    for key in averaged_weights:
        averaged_weights[key] /= num_models

    averaged_model = model_class.from_pretrained(model_paths[0], device_map=device_map)
    averaged_model.load_state_dict(averaged_weights)

    return averaged_model


def merge_models(model_paths, output_path=None, skip_merged=False):
    device = 'cpu' # we're not running the models, so we can load on CPU

    if output_path is None:
        # Default to the same directory
        output_path = os.path.dirname(model_paths[0])
        output_path = os.path.join(output_path, f"last-{len(model_paths)}-model-merged")

    if skip_merged and os.path.exists(output_path):
        print(f"Model found at {output_path}. Skipping...")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_paths[-1], device_map=device)

    merged_model = weightspace_average_model(
        model_paths,
        AutoModelForCausalLM,
        device_map=device
    )
    os.makedirs(output_path, exist_ok=True)

    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


def merge_from_file(args):
    model_paths = load_model_paths(args.model_list_file)
    print(f'Running merging for models: {model_paths}')
    merge_models(model_paths)


def merge_from_list(model_list):
    # Use python lists for merging
    unique_model_list = list(set(['/'.join(model.split('/')[:-1]) for model in model_list]))
    all_model_list = model_list

    for idx, unique_model in enumerate(unique_model_list):
        print(f"({idx}/{len(unique_model_list)}) Running merging for {unique_model}")

        model_paths = [model for model in all_model_list if unique_model in model]
        model_paths = [model.replace("weka://", '/') for model in model_paths]

        print(f'{unique_model}/last-{len(model_paths)}-model-merged')
        
        merge_models(model_paths, skip_merged=True)


def merge_last_n(checkpoint_list):
    # Special last-n merging experiment
    for idx in range(len(checkpoint_list)):
        print(f"({idx}/{len(checkpoint_list)}) Running merging for final {idx} checkpoints")

        model_paths = checkpoint_list[-(idx+1):]

        output_path = os.path.dirname(os.path.dirname(model_paths[0]))
        output_path = os.path.join(output_path, 'prox_fineweb_pro_merged-1B-5xC-2', f"last-{len(model_paths)}-model-merged")
        
        merge_models(model_paths, output_path=output_path, skip_merged=True)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Weightspace averaging of models.")
    # parser.add_argument(
    #     "--model-list-file",
    #     type=str,
    #     required=True,
    #     help="Path to the file containing the list of model paths."
    # )
    # args = parser.parse_args()
    # merge_from_list(args)

    # merge_from_list(MODEL_LIST_FINAL_SIX_CKPTS + DATADECIDE_FINAL_FIVE_CKPTS)

    """
    models: 
    /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro_merged-1B-5xC-2

    oe-eval --model /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro_merged-1B-5xC-2/last-1-model-merged --task arc_easy:rc::olmes:full arc_challenge:rc::olmes:full --run-local --output-dir workspace --batch-size 128

    - ARC-E: 0.722
    - ARC-C: 0.431

    oe-eval --model /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro_merged-1B-5xC-2/last-2-model-merged --task arc_easy:rc::olmes:full arc_challenge:rc::olmes:full --run-local --output-dir workspace --batch-size 128

    - ARC-E: 0.720
    - ARC-C: 0.433

    oe-eval --model /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro_merged-1B-5xC-2/last-8-model-merged --task arc_easy:rc::olmes:full arc_challenge:rc::olmes:full --run-local --output-dir workspace --batch-size 128

    - ARC-E: 0.723
    - ARC-C: 0.438

    oe-eval --model /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro_merged-1B-5xC-2/last-20-model-merged --task arc_easy:rc::olmes:full arc_challenge:rc::olmes:full --run-local --output-dir workspace --batch-size 128

    - ARC-E: 0.724
    - ARC-C: 0.430
    """
    merge_last_n(FINEWEB_CKPTS)
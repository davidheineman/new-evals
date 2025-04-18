import os, time, sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from analysis.utils import weka_to_gcs
from analysis.utils.constants_models import MODEL_LADDER_LIST, MODEL_LIST_MIXES_FINAL, MODEL_LIST_MIXES_FINAL_EXTENDED, MODEL_LIST_INTERMEDIATE, MODEL_LIST_INTERMEDIATE_7B, MODEL_LIST_INTERMEDIATE_13B, MODEL_LIST_INTERMEDIATE_32B, MODEL_LIST_MIXES, OE_EVAL_BASE_MODELS, OE_EVAL_INSTRUCT_MODELS, OE_EVAL_ALL_MODELS, OE_EVAL_BASE_MODELS_EXTENDED

MODEL_LIST_ALL = []
# MODEL_LIST_ALL += MODEL_LADDER_LIST
MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE
MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE_7B
MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE_13B # 13B intermediate ckpts
MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE_32B
MODEL_LIST_ALL += MODEL_LIST_MIXES_FINAL + MODEL_LIST_MIXES_FINAL_EXTENDED # DataDecide

# Exact paths of model ladder
MODEL_LIST_ALL += [
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-0.5xC/step3641-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-10xC/step72625-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-2xC/step14533-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-5xC/step36318-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-0.5xC/step8145-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-10xC/step162000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-2xC/step32547-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-370M-0.5xC/step4731-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-370M-10xC/step94427-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-370M-2xC/step18894-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-370M-5xC/step47219-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-3B-0.5xC/step10086-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-3B-10xC/step201524-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-3B-2xC/step40313-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-3B-5xC/step100767-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-760M-0.5xC/step5795-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-760M-10xC/step115706-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-760M-2xC/step23150-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-760M-5xC/step57858-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-190M-1xC/step7272-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-1B-1xC/step16279-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-370M-1xC/step9452-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-3B-1xC/step20162-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-760M-1xC/step11580-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7/step928646-hf-vllm-2", # converted to new vllm format
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step476848-hf-vllm",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81352-unsharded-hf"
]

# Filter for weka models and extract paths after checkpoints/
weka_models = [model for model in MODEL_LIST_ALL if model.startswith('weka://')]
gcs_paths = [weka_to_gcs(model) for model in weka_models]

# Create rsync commands
from concurrent.futures import ThreadPoolExecutor

def check_and_sync(weka_model, gcs_path):
    weka_model = weka_model.replace('weka://', '/')
    check_command = f"gsutil ls {gcs_path}"
    
    if os.system(check_command) != 0:
        command = f"""\
            gsutil -o "GSUtil:parallel_composite_upload_threshold=150M" -m rsync -r {weka_model} {gcs_path} \
        """
        print(f'Executing command:\n{command}')
        os.system(command)
    else:
        print(f'Skipping {gcs_path} - already exists')

with ThreadPoolExecutor() as executor:
    executor.map(lambda x: check_and_sync(*x), zip(weka_models, gcs_paths))

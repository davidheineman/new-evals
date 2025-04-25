import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

WEKA_PATH = "/data/input/"

def load_model_paths(file_path):
    """
    Load model paths from a file.

    Args:
        file_path (str): Path to the file containing model paths.

    Returns:
        list: List of model paths.
    """
    with open(file_path, "r") as f:
        return [line.strip().replace("weka://oe-eval-default/", WEKA_PATH) for line in f if line.strip()]


def weightspace_average_model(model_paths, model_class, device_map="auto"):
    """
    Perform weightspace averaging of models.

    Args:
        model_paths (list): List of model paths.
        model_class: The class of the model to load (e.g., `AutoModelForCausalLM`).
        device_map (str): Device map for loading models.

    Returns:
        Averaged model.
    """
    averaged_weights = None
    num_models = len(model_paths)

    for path in model_paths:
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


def main(args):
    model_paths = load_model_paths(args.model_list_file)
    tokenizer = AutoTokenizer.from_pretrained(model_paths[-1], device_map="auto")

    merged_model = weightspace_average_model(
        model_paths,
        AutoModelForCausalLM,
        device_map="auto"
    )

    output_path = os.path.dirname(model_paths[0])
    output_path = os.path.join(output_path, f"last-{len(model_paths)}-model-merged")
    os.makedirs(output_path, exist_ok=True)

    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weightspace averaging of models.")
    parser.add_argument(
        "--model-list-file",
        type=str,
        required=True,
        help="Path to the file containing the list of model paths."
    )
    args = parser.parse_args()
    main(args)
